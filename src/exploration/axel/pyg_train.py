import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
import random
import numpy as np
from torch_geometric.data import HeteroData
from util.graphs import FullGraph
from tqdm import tqdm

curr_dir = os.path.dirname(__file__)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ----------------- Lightweight GNN Model for M1 MacBook Air -----------------
class LightweightUserItemGNN(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim=64, num_layers=2, dropout=0.2):
        """Smaller model that fits in M1 MacBook Air memory"""
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items, hidden_dim)
        
        # Fewer layers
        self.user_convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_dim) for _ in range(num_layers)
        ])
        self.item_convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_dim) for _ in range(num_layers)
        ])
        
        # Batch normalization for stability
        self.user_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.item_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = dropout
        
        # Simpler prediction head
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, edge_index, edge_label_index):
        """
        Args:
            edge_index: Graph structure edges (only positive edges)
            edge_label_index: Edges to predict on (pos + neg)
        """
        user_x = self.user_emb.weight
        item_x = self.item_emb.weight
        
        # Multi-layer message passing
        for i, (user_conv, item_conv, user_bn, item_bn) in enumerate(
            zip(self.user_convs, self.item_convs, self.user_bns, self.item_bns)
        ):
            # Message passing: aggregate item info into users
            user_x_new = user_conv((item_x, user_x), edge_index.flip(0))
            user_x_new = user_bn(user_x_new)
            user_x_new = F.relu(user_x_new)
            user_x_new = F.dropout(user_x_new, p=self.dropout, training=self.training)
            
            # Message passing: aggregate user info into items
            item_x_new = item_conv((user_x, item_x), edge_index)
            item_x_new = item_bn(item_x_new)
            item_x_new = F.relu(item_x_new)
            item_x_new = F.dropout(item_x_new, p=self.dropout, training=self.training)
            
            # Simple update (no residual to save memory)
            user_x = user_x_new
            item_x = item_x_new
        
        # Get embeddings for prediction edges
        user_ids, item_ids = edge_label_index
        u_h = user_x[user_ids]
        i_h = item_x[item_ids]
        x = torch.cat([u_h, i_h], dim=1)
        
        return self.fc(x)

# ----------------- Mini-batch Training Pipeline -----------------
def train_pipeline_minibatch(data, num_users, num_items, epochs=100, 
                             batch_size=8192, hidden_dim=64, num_layers=2, 
                             lr=1e-3, weight_decay=1e-5):
    """Memory-efficient training with mini-batches"""
    
    # Check for MPS (Metal Performance Shaders) on M1
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Initialize lightweight model
    model = LightweightUserItemGNN(num_users, num_items, hidden_dim=hidden_dim, 
                                   num_layers=num_layers).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
    
    # Use ONLY positive edges for message passing (graph structure)
    edge_index = data['user', 'interacts', 'item'].edge_index.to(device)
    
    # Use ALL edges (pos + neg) for prediction
    edge_label_index = data['user', 'interacts', 'item'].edge_label_index
    edge_label = data['user', 'interacts', 'item'].edge_label
    
    print(f"\nDataset statistics:")
    print(f"  Graph structure edges (positive only): {edge_index.size(1):,}")
    print(f"  Prediction edges (pos + neg): {edge_label_index.size(1):,}")
    
    # Analyze class distribution
    class_counts = torch.bincount(edge_label)
    print(f"\nClass distribution:")
    class_names = ["No Interaction", "View", "Save", "Buy"]
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        pct = 100 * count / len(edge_label)
        print(f"  Class {i} ({name:15s}): {count:7d} ({pct:5.2f}%)")
    
    # Split edges for training/validation (80/20)
    n_edges = edge_label.size(0)
    n_train = int(0.8 * n_edges)
    perm = torch.randperm(n_edges)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    # Calculate class weights
    weights = 1.0 / (class_counts.float() + 1e-6)
    weights = weights ** 0.5  # Square root for softer weighting
    weights = weights / weights.sum() * len(weights)
    print(f"\nClass weights: {weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    print("\nStarting training...")
    
    # Initialize model with dummy forward pass
    model.eval()
    with torch.no_grad():
        dummy_edges = edge_label_index[:, :10].to(device)
        _ = model(edge_index, dummy_edges)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    print(f"Batch size: {batch_size:,} edges per batch\n")
    
    # Training loop with mini-batches
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(epochs):
        # Training with mini-batches
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle training indices
        train_perm = train_idx[torch.randperm(len(train_idx))]
        
        for i in range(0, len(train_perm), batch_size):
            batch_idx = train_perm[i:i+batch_size]
            
            # Get batch data
            batch_edge_label_index = edge_label_index[:, batch_idx].to(device)
            batch_labels = edge_label[batch_idx].to(device)
            
            optimizer.zero_grad()
            logits = model(edge_index, batch_edge_label_index)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation with mini-batches (to save memory)
        model.eval()
        val_preds_list = []
        
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size * 2):  # Larger batches for inference
                batch_idx = val_idx[i:i+batch_size*2]
                batch_edge_label_index = edge_label_index[:, batch_idx].to(device)
                
                logits = model(edge_index, batch_edge_label_index)
                preds = logits.argmax(dim=1).cpu()
                val_preds_list.append(preds)
        
        val_preds = torch.cat(val_preds_list)
        val_labels = edge_label[val_idx]
        
        # Accuracy
        val_acc = (val_preds == val_labels).float().mean().item()
        
        # Per-class metrics for Macro F1
        TP = torch.zeros(4)
        FP = torch.zeros(4)
        FN = torch.zeros(4)
        
        for i in range(4):
            TP[i] = ((val_preds == i) & (val_labels == i)).sum()
            FP[i] = ((val_preds == i) & (val_labels != i)).sum()
            FN[i] = ((val_preds != i) & (val_labels == i)).sum()
        
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        macro_f1 = f1.mean().item()
        
        # Learning rate scheduling
        scheduler.step(macro_f1)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        # Early stopping based on F1
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(curr_dir, "user_item_gnn_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': macro_f1,
                'class_weights': weights.cpu(),
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
            }, save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation Macro-F1: {best_val_f1:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")
    
    return model

# ----------------- Run -----------------
if __name__ == '__main__':
    print("Loading training data...")
    G = FullGraph(os.path.join(curr_dir, "../../data/task1_train.tsv"))
    
    print(f"\nGraph statistics:")
    print(f"  Users: {len(G.user_to_items):,}")
    print(f"  Items: {len(G.item_to_users):,}")
    print(f"  Positive interactions: {len(G.user_item_to_interaction):,}")
    
    # Use LOWER negative sampling ratio for memory efficiency
    # Still much better than the original 3:1
    negative_ratio = 5  # Reduced from 10 to fit in memory
    print(f"\nBuilding graph with negative sampling ratio: {negative_ratio}:1")
    print("(Reduced for memory efficiency on M1 MacBook Air)")
    
    data, user2idx, item2idx = G.build_pyg_graph(num_negative_samples_per_pos=negative_ratio)
    
    print("\nStarting training pipeline...")
    model = train_pipeline_minibatch(
        data, 
        len(user2idx), 
        len(item2idx),
        epochs=100,
        batch_size=8192,  # Process 8K edges at a time
        hidden_dim=64,    # Smaller than 128
        num_layers=2,     # Fewer than 3
        lr=1e-3,
        weight_decay=1e-5
    )

