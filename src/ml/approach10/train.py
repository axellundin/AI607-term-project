import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # tested this, but does not work. 

from ml.approach10.model import HeteroLightGCN
from ml.approach10.data import load_dataset, load_validation_dataset, get_negative_samples
from torch_geometric.loader import LinkNeighborLoader
from settings import *
import torch 
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# Define hyperparameters 
num_epochs = 50
embedding_dim = 256
hidden_channels = 256
num_layers = 2
batch_size = 8192 * 2
learning_rate = 0.005
dropout_edge_prob = 0.3  # Edge dropout for regularization
weight_decay = 1e-4  # L2 regularization

# Load dataset 
data, user2idx, item2idx, labels = load_dataset(training_data_filename)

# Add negative samples
num_negative_samples = len(labels) 
print(f"Generating {num_negative_samples} negative samples...")
negative_labels = get_negative_samples(labels, user2idx, item2idx, num_negative_samples)
num_users = len(user2idx)
num_items = len(item2idx)

labels.update(negative_labels)

train_pairs = list(labels.keys())  
train_labels = [labels[pair] for pair in train_pairs] 

# Load validation set 
val_data_dict = load_validation_dataset(val_data_filename)
val_pairs = list(val_data_dict.keys())
val_labels = [val_data_dict[pair] for pair in val_pairs]

# Initiate model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroLightGCN(num_users, num_items, embedding_dim, hidden_channels, num_layers=num_layers).to(device)

data = data.to(device)

# Ordinal regression loss function
def ordinal_regression_loss(logits, targets):
    """
    Compute ordinal regression loss as sum of elementwise binary cross-entropy.
    
    Args:
        logits: Tensor of shape (batch_size, 3) - 3 logits for view, save, buy
        targets: Tensor of shape (batch_size, 3) - ordinal encoding: (0,0,0), (1,0,0), (1,1,0), (1,1,1)
    
    Returns:
        loss: Scalar tensor
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Compute binary cross-entropy for each logit
    bce1 = F.binary_cross_entropy(probs[:, 0], targets[:, 0].float())
    bce2 = F.binary_cross_entropy(probs[:, 1], targets[:, 1].float()) * 1.5
    bce3 = F.binary_cross_entropy(probs[:, 2], targets[:, 2].float())
    
    # Sum the three BCE losses
    loss = bce1 + bce2 + bce3
    return loss

# Function to encode labels to ordinal format
def encode_ordinal_labels(labels):
    """
    Convert class labels to ordinal encoding:
    - no_interaction (0) → (0, 0, 0)
    - view (1) → (1, 0, 0)
    - save (2) → (1, 1, 0)
    - buy (3) → (1, 1, 1)
    """
    ordinal_labels = torch.zeros(len(labels), 3, dtype=torch.long)
    for i, label in enumerate(labels):
        if label == 1:  # view
            ordinal_labels[i] = torch.tensor([1, 0, 0])
        elif label == 2:  # save
            ordinal_labels[i] = torch.tensor([1, 1, 0])
        elif label == 3:  # buy
            ordinal_labels[i] = torch.tensor([1, 1, 1])
        # label == 0 (no_interaction) remains (0, 0, 0)
    return ordinal_labels

# Function to predict from logits (for accuracy calculation)
def predict_with_threshold(logits, threshold=0.5):
    """Convert logits to class predictions using threshold."""
    probs = torch.sigmoid(logits)  # shape (batch, 3)
    b0 = probs[:, 0] > threshold   # predicts y > 0
    b1 = probs[:, 1] > threshold   # predicts y > 1
    b2 = probs[:, 2] > threshold   # predicts y > 2
    pred = b0.int() + b1.int() + b2.int()
    return pred

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Checkpoint loading
checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach10.pt")
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Resume from the saved epoch (or start from next epoch)
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"Resuming training from epoch {start_epoch + 1}")
    print(f"Model and optimizer states loaded successfully")
else:
    print("No checkpoint found. Starting training from scratch.")

print("Starting Training!")
print("=" * 60)

last_epoch = start_epoch
for epoch in range(start_epoch, num_epochs):
    model.train()
    
    # Shuffle training data
    perm = torch.randperm(len(train_pairs))

    # Edge Dropout for regularization - drop edges in the message passing graph
    # Store original edge indices for view, save, buy edges
    original_edges = {}
    
    # Apply edge dropout to each interaction type (view, save, buy)
    for edge_name, rev_name in [('view', 'viewed_by'), ('save', 'saved_by'), ('buy', 'bought_by')]:
        edge_type = ('user', edge_name, 'item')
        rev_edge_type = ('item', rev_name, 'user')
        
        # Store and drop forward edges
        if data[edge_type].edge_index.size(1) > 0:
            original_edges[edge_type] = data[edge_type].edge_index.clone()
            mask = torch.rand(original_edges[edge_type].size(1), device=device) > dropout_edge_prob
            data[edge_type].edge_index = original_edges[edge_type][:, mask]
        
        # Store and drop reverse edges
        if data[rev_edge_type].edge_index.size(1) > 0:
            original_edges[rev_edge_type] = data[rev_edge_type].edge_index.clone()
            rev_mask = torch.rand(original_edges[rev_edge_type].size(1), device=device) > dropout_edge_prob
            data[rev_edge_type].edge_index = original_edges[rev_edge_type][:, rev_mask]
    
    total_loss = 0
    num_batches = 0
    correct_predictions = 0
    total_samples = 0
    
    # Batch training
    for i in tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_indices = perm[i:i+batch_size]
        
        # Get batch data
        batch_pairs = [train_pairs[idx] for idx in batch_indices]
        batch_labels_raw = [train_labels[idx] for idx in batch_indices]
        
        # Convert IDs to indices
        batch_user_ids = torch.tensor([user2idx[user_id] for user_id, _ in batch_pairs], device=device)
        batch_item_ids = torch.tensor([item2idx[item_id] for _, item_id in batch_pairs], device=device)
        
        # Encode labels to ordinal format
        batch_labels_ordinal = encode_ordinal_labels(batch_labels_raw).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(data, batch_user_ids, batch_item_ids)  # Shape: (batch_size, 3)
        loss = ordinal_regression_loss(logits, batch_labels_ordinal)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate accuracy (minimal overhead - reuse logits)
        with torch.no_grad():
            preds = predict_with_threshold(logits, threshold=0.5)
            batch_labels_tensor = torch.tensor(batch_labels_raw, device=device)
            correct_predictions += (preds == batch_labels_tensor).sum().item()
            total_samples += len(batch_labels_raw)
        
        total_loss += loss.item()
        num_batches += 1
    
    # Restore original edges after epoch
    for edge_type, original_edge_index in original_edges.items():
        if hasattr(data, edge_type[0]) and hasattr(data[edge_type[0], edge_type[1], edge_type[2]], 'edge_index'):
            data[edge_type].edge_index = original_edge_index

    avg_loss = total_loss / num_batches
    train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    last_epoch = epoch + 1
    # START DEBUG---
    # Check gradients
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"  Gradient norm: {total_norm:.4f}")
    
    # Check parameter values
    for name, param in model.named_parameters():
        if 'attention' in name and param.requires_grad:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, max={param.data.max():.4f}")
    
    # Check edge counts
    print(f"  View edges: {data['user', 'view', 'item'].edge_index.size(1)}")
    print(f"  Save edges: {data['user', 'save', 'item'].edge_index.size(1)}")
    print(f"  Buy edges: {data['user', 'buy', 'item'].edge_index.size(1)}")
    # END DEBUG---

    # Save checkpoint periodically
    checkpoint_interval = 1
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach10.pt")
        torch.save({
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'user2idx': user2idx,
            'item2idx': item2idx,
            'hyperparameters': {
                'num_users': num_users,
                'num_items': num_items,
                'embedding_dim': embedding_dim,
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
            }
        }, checkpoint_path)
        print(f"  Checkpoint saved at epoch {epoch + 1}")

print("=" * 60)
print("Training complete!")

# Save model checkpoint
save_path = os.path.join(models_dir, "hetero_sage_model_approach10.pt")
torch.save({
    'epoch': last_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'user2idx': user2idx,
    'item2idx': item2idx,
    'hyperparameters': {
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': embedding_dim,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
    }
}, save_path)
print(f"Model checkpoint saved to {save_path}")

