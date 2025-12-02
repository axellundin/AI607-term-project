import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # tested this, but does not work. 

from ml.approach5.model import HeteroSAGE
from ml.approach5.data import load_dataset, load_validation_dataset, get_negative_samples
from torch_geometric.loader import LinkNeighborLoader
from settings import *
import torch 
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# Define hyperparamters 
num_epochs = 23
# embedding_dim = 128
# hidden_channels = 64
embedding_dim = 256
hidden_channels = 128
batch_size = 8192 * 2
learning_rate = 0.01
dropout_edge_prob = 0.3 # Probability to drop edges in the graph structure during training

# Load dataset 
data, user2idx, item2idx, labels = load_dataset(training_data_filename)

# Add negative samples
num_negative_samples = len(labels) 
print(f"Generating {num_negative_samples} negative samples...")
# Optimized negative sampling could go here, but using existing function for now
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
model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
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
    bce2 = F.binary_cross_entropy(probs[:, 1], targets[:, 1].float())
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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Added weight decay

# Checkpoint loading
checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach5.pt")  # Updated path for approach5
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
    
    # Optionally load other saved data
    # user2idx = checkpoint.get('user2idx', user2idx)
    # item2idx = checkpoint.get('item2idx', item2idx)
    
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

    # Edge Dropout for 'interact' edges to prevent overfitting to edge existence
    # We only drop edges in the message passing graph, NOT in the supervision labels.
    original_edge_index = data['user', 'interact', 'item'].edge_index
    original_rev_edge_index = data['item', 'interact_by', 'user'].edge_index
    
    # Create a mask
    mask = torch.rand(original_edge_index.size(1), device=device) > dropout_edge_prob
    
    # Apply mask temporarily
    data['user', 'interact', 'item'].edge_index = original_edge_index[:, mask]
    # Ideally we should also drop the corresponding reverse edges, but independent dropout is also a valid regularization.
    # For strict correctness, let's drop reverse edges independently or sync them. Independent is fine for regularization.
    rev_mask = torch.rand(original_rev_edge_index.size(1), device=device) > dropout_edge_prob
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index[:, rev_mask]
    
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
    
    # Restore original edges after epoch (or after batch if you prefer, but per-epoch is faster)
    data['user', 'interact', 'item'].edge_index = original_edge_index
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index

    avg_loss = total_loss / num_batches
    train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    last_epoch = epoch + 1
    # Save checkpoint periodically (e.g., every 10 epochs)
    checkpoint_interval = 10
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach5.pt")
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
            }
        }, checkpoint_path)
        print(f"  Checkpoint saved at epoch {epoch + 1}")

print("=" * 60)
print("Training complete!")

# Save model checkpoint
save_path = os.path.join(models_dir, "hetero_sage_model_approach5.pt")
torch.save({
    'epoch': last_epoch,  # Save the last completed epoch
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

