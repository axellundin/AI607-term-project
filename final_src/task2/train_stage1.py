import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # tested this, but does not work. 

from task2.model_stage1 import HeteroSAGE
from task2.data_stage1 import load_combined_dataset, load_validation_dataset, get_negative_samples
from torch_geometric.loader import LinkNeighborLoader
from settings import *
import torch 
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--embedding_dim", type=int, default=256)
parser.add_argument("--hidden_channels", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--dropout_edge_prob", type=float, default=0.1)
parser.add_argument("--model_name", type=str, default="task2_stage1.pt")
args = parser.parse_args()


# Define hyperparamters 
num_epochs = args.num_epochs
embedding_dim = args.embedding_dim
hidden_channels = args.hidden_channels
batch_size = args.batch_size
learning_rate = args.learning_rate
dropout_edge_prob = args.dropout_edge_prob # Probability to drop edges in the graph structure during training

# Load combined dataset from both task1 and task2
print("Loading combined dataset from task1 and task2...")
data, user2idx, item2idx, labels, all_interactions = load_combined_dataset(
    task1_filename="task1_train.tsv",
    task2_filename="task2_train.tsv"
)
print(f"Loaded {len(user2idx)} users (task1 + task2), {len(item2idx)} items")
print(f"Training labels (save/buy only): {len(labels)}")

# Add negative samples (exclude any interaction from either task)
num_negative_samples = len(labels) 
print(f"Generating {num_negative_samples} negative samples...")
negative_labels = get_negative_samples(all_interactions, user2idx, item2idx, num_negative_samples)
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
        logits: Tensor of shape (batch_size, 2) - 2 logits for save, buy
        targets: Tensor of shape (batch_size, 2) - ordinal encoding: (0,0), (1,0), (1,1)
    
    Returns:
        loss: Scalar tensor
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Compute binary cross-entropy for each logit
    bce_save = F.binary_cross_entropy(probs[:, 0], targets[:, 0].float())
    bce_buy = F.binary_cross_entropy(probs[:, 1], targets[:, 1].float())
    
    # Sum the two BCE losses
    loss = bce_save + bce_buy
    return loss

# Function to encode labels to ordinal format
def encode_ordinal_labels(labels):
    """
    Convert class labels to ordinal encoding for 2 logits (save, buy):
    - no_interaction (0) → (0, 0)
    - save (2) → (1, 0)
    - buy (3) → (1, 1)
    Note: view (1) interactions are not used for training
    """
    ordinal_labels = torch.zeros(len(labels), 2, dtype=torch.long)
    for i, label in enumerate(labels):
        if label == 2:  # save
            ordinal_labels[i] = torch.tensor([1, 0])
        elif label == 3:  # buy
            ordinal_labels[i] = torch.tensor([1, 1])
        # label == 0 (no_interaction) remains (0, 0)
    return ordinal_labels

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Added weight decay

# Checkpoint loading
checkpoint_path = os.path.join(models_dir, args.model_name)
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
        
        total_loss += loss.item()
        num_batches += 1
    
    # Restore original edges after epoch (or after batch if you prefer, but per-epoch is faster)
    data['user', 'interact', 'item'].edge_index = original_edge_index
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index

    avg_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")
    last_epoch = epoch + 1
    # Save checkpoint periodically (e.g., every 10 epochs)
    checkpoint_interval = 10
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(models_dir, args.model_name)
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
save_path = os.path.join(models_dir,args.model_name)
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

