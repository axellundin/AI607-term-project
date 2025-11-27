import os
import sys

from torch.cpu import is_available
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Adjust path so imports work if run from subfolder (optional, but good practice)
sys.path.append(os.getcwd())
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from ml.task2_approach3.model import HeteroSAGE
from ml.task2_approach3.data import load_dataset, load_validation_dataset
import torch 
from tqdm import tqdm
import numpy as np

def train():
    training_data_filename = "task2_train.tsv"
    val_data_filename = "task2_val_answers.tsv"
    # Define hyperparamters 
    num_epochs = 3
    embedding_dim = 128
    hidden_channels = 64
    batch_size = 4096 # Smaller batch size for BPR (pairs vs triplets)
    learning_rate = 0.01

    # Load dataset 
    data, user2idx, item2idx, labels = load_dataset(training_data_filename)

    # Only use positive interactions
    # interaction types 2 and 3 are positive (save and buy)
    # data.py already puts them in labels. 
    
    # Filter to get only positive pairs (keys where value > 0)
    # Actually labels contains interaction type, which is > 0.
    # load_dataset in data.py: labels[(user_id, item_id)] = interaction (2 or 3)
    
    pos_pairs = list(labels.keys())
    print(f"Total positive training samples: {len(pos_pairs)}")

    num_users = len(user2idx)
    num_items = len(item2idx)

    # Initiate model 
    if torch.cuda.is_available(): 
        device="cuda"
    else : device="cpu"

    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    data = data.to(device)

    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Checkpoint loading
    new_models_dir = "src/results/models_task2_approach4"
    if not os.path.exists(new_models_dir):
        os.makedirs(new_models_dir)
        
    checkpoint_path = os.path.join(new_models_dir, "hetero_sage_model.pt") 
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    print("Starting BPR Training!")
    print("=" * 60)

    last_epoch = start_epoch
    
    # Pre-convert all user/item strings to indices for faster access
    all_user_indices = [user2idx[u] for u, _ in pos_pairs]
    all_item_indices = [item2idx[i] for _, i in pos_pairs]
    
    all_user_indices = torch.tensor(all_user_indices, dtype=torch.long)
    all_item_indices = torch.tensor(all_item_indices, dtype=torch.long)
    
    # Edge types to apply dropout to
    edge_types_to_drop = [
        ('user', 'save', 'item'),
        ('user', 'buy', 'item'),
        ('item', 'saved_by', 'user'),
        ('item', 'bought_by', 'user')
    ]
    dropout_edge_prob = 0.5

    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        # ---------------------------------------------------------
        # Apply Edge Dropout to prevent overfitting/leakage
        # ---------------------------------------------------------
        original_edges = {}
        for etype in edge_types_to_drop:
            if etype in data.edge_index_dict:
                original_edges[etype] = data[etype].edge_index
                e_index = data[etype].edge_index
                mask = torch.rand(e_index.size(1), device=device) > dropout_edge_prob
                data[etype].edge_index = e_index[:, mask]
        
        # Shuffle training data indices
        perm = torch.randperm(len(pos_pairs))
        
        total_loss = 0
        num_batches = 0
        
        # Batch training
        for i in tqdm(range(0, len(pos_pairs), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_indices = perm[i:i+batch_size]
            
            # Get positive batch
            batch_user_ids = all_user_indices[batch_indices].to(device)
            batch_pos_item_ids = all_item_indices[batch_indices].to(device)
            
            current_batch_size = batch_user_ids.size(0)
            
            # Dynamic Negative Sampling: Sample random item indices
            # We don't strictly check for collisions (false negatives) for speed, 
            # assuming sparsity makes collisions rare.
            batch_neg_item_ids = torch.randint(0, num_items, (current_batch_size,), device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            # Forward pass for positives and negatives
            # HeteroSAGE forward returns (batch_size, 1) scores
            pos_scores = model(data, batch_user_ids, batch_pos_item_ids).squeeze()
            neg_scores = model(data, batch_user_ids, batch_neg_item_ids).squeeze()
            
            # BPR Loss
            # loss = - log(sigmoid(pos_score - neg_score))
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # ---------------------------------------------------------
        # Restore original edges
        # ---------------------------------------------------------
        for etype, edge_index in original_edges.items():
            data[etype].edge_index = edge_index

        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | BPR Loss: {avg_loss:.4f}")
        last_epoch = epoch + 1
        
        checkpoint_interval = 5
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, user2idx, item2idx, last_epoch, num_users, num_items, embedding_dim, hidden_channels, new_models_dir)

    print("=" * 60)
    print("Training complete!")

    save_checkpoint(model, optimizer, user2idx, item2idx, last_epoch, num_users, num_items, embedding_dim, hidden_channels, new_models_dir)

def save_checkpoint(model, optimizer, user2idx, item2idx, epoch, num_users, num_items, embedding_dim, hidden_channels, _models_dir):
    save_path = os.path.join(_models_dir, "hetero_sage_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'user2idx': user2idx,
        'item2idx': item2idx,
        'hyperparameters': {
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': embedding_dim,
            'hidden_channels': hidden_channels,
        }
    }, save_path)
    print(f"Model checkpoint saved to {save_path}")

if __name__ == "__main__" : 
    train()
