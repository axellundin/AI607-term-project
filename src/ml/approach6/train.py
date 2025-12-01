import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

from ml.approach6.model import HeteroSAGE
from ml.approach6.data import load_combined_dataset, load_validation_dataset, get_negative_samples
from torch_geometric.loader import LinkNeighborLoader
from settings import *
import torch 
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

# --- Configuration Flags ---
USE_TASK2_NEG_SAMPLING = True # Toggle negative sampling for Task 2 users

# Define hyperparamters 
num_epochs = 18 # Increased from 2 to 10
embedding_dim = 256
hidden_channels = 128
batch_size = 8192 * 2
learning_rate = 0.01
dropout_edge_prob = 0.4 # Probability to drop edges in the graph structure during training

# Load combined dataset 
print("Loading combined dataset...")
data, user2idx, item2idx, labels, task1_users, task2_users = load_combined_dataset(
    task1_filename="task1_train.tsv",
    task2_filename="task2_train.tsv"
)
print(f"Loaded {len(user2idx)} users ({len(task1_users)} task1, {len(task2_users)} task2), {len(item2idx)} items")

# Add negative samples
num_negative_samples = len(labels) 
print(f"Generating {num_negative_samples} negative samples (Task 2 neg sampling: {USE_TASK2_NEG_SAMPLING})...")

negative_labels = get_negative_samples(
    labels, 
    user2idx, 
    item2idx, 
    num_negative_samples,
    task1_users,
    task2_users,
    use_task2_neg_sampling=USE_TASK2_NEG_SAMPLING
)

num_users = len(user2idx)
num_items = len(item2idx)

labels.update(negative_labels)

train_pairs = list(labels.keys())  
train_labels = [labels[pair] for pair in train_pairs] 

# Calculate Class Weights for CORN
print("\nCalculating class weights for CORN loss...")
label_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=4)
print(f"Label counts: {label_counts.tolist()}")

# Task 0: y > 0 vs y <= 0 (Neg: 0, Pos: 1,2,3)
# Usually balanced by neg sampling, but let's check
pos_0 = label_counts[1:].sum().item()
neg_0 = label_counts[0].item()
weight_0 = neg_0 / pos_0 if pos_0 > 0 else 1.0

# Task 1: y > 1 vs y <= 1 (cond y >= 1) -> Neg: 1, Pos: 2,3
pos_1 = label_counts[2:].sum().item()
neg_1 = label_counts[1].item()
weight_1 = neg_1 / pos_1 if pos_1 > 0 else 1.0

# Task 2: y > 2 vs y <= 2 (cond y >= 2) -> Neg: 2, Pos: 3
pos_2 = label_counts[3].item()
neg_2 = label_counts[2].item()
weight_2 = neg_2 / pos_2 if pos_2 > 0 else 1.0

task_weights = [weight_0, weight_1, weight_2]
print(f"Computed Task Weights (Pos Weight): {task_weights}")
# We will use these as pos_weight in BCEWithLogitsLoss

# Create Edge Index for Supervision
# We need to create a temporary edge_index to use with LinkNeighborLoader
# This edge_index represents the supervision edges (user, supervise, item)
# We map the list of (user, item) pairs to an edge_index
user_indices = [user2idx[u] for u, i in train_pairs]
item_indices = [item2idx[i] for u, i in train_pairs]
supervision_edge_index = torch.tensor([user_indices, item_indices], dtype=torch.long)

# Add supervision edges to data object so LinkNeighborLoader can sample from them
# We use a dummy relation type ('user', 'supervision', 'item')
data['user', 'supervision', 'item'].edge_index = supervision_edge_index
# Store labels as edge attributes or a separate tensor aligned with edge_index
# Here we'll just keep them in a list and access by edge_label_index order? 
# Better: LinkNeighborLoader returns edge_label_index which are indices into the original edge_index
# So we can index into our labels tensor.
supervision_labels = torch.tensor(train_labels, dtype=torch.long)

# Load validation set 
val_data_dict = load_validation_dataset(val_data_filename)
val_pairs = list(val_data_dict.keys())
val_labels = [val_data_dict[pair] for pair in val_pairs]

# Initiate model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
data = data.to(device)
supervision_labels = supervision_labels.to(device)

# Initialize LinkNeighborLoader
# We sample neighbors for the 'interact' and 'interact_by' edges
# We start sampling from the supervision edges
train_loader = LinkNeighborLoader(
    data,
    num_neighbors=[10, 5], # Sample 10 neighbors at 1st hop, 5 at 2nd hop
    edge_label_index=(('user', 'supervision', 'item'), supervision_edge_index),
    edge_label=supervision_labels,
    batch_size=batch_size,
    shuffle=True,
    neg_sampling_ratio=0.0, # We already added negative samples manually to the set
    num_workers=0 # Set to >0 for multi-processing if supported
)

# CORN Loss Function
def corn_loss(logits, targets, task_weights, num_classes=4):
    """
    Compute CORN loss (Conditional Ordinal Regression) with class weighting.
    
    Args:
        logits: Tensor of shape (batch_size, K-1) 
                Logit k corresponds to P(y > k | y >= k)
        targets: Tensor of shape (batch_size,) with integer labels 0..K-1
        task_weights: List of K-1 float weights for positive class in each task
        num_classes: K (default 4 for 0,1,2,3)
    
    Returns:
        loss: Scalar tensor
    """
    loss = 0
    # We have K-1 binary tasks
    # Task k (0 to K-2): Predict if y > k given y >= k
    for k in range(num_classes - 1):
        # Filter: consider only samples where y >= k
        mask = targets >= k
        if mask.sum() > 0:
            # For these samples, label is 1 if y > k, else 0
            binary_targets = (targets[mask] > k).float()
            preds = logits[mask, k]
            
            # Apply pos_weight for this task
            pos_weight = torch.tensor(task_weights[k], device=logits.device)
            
            loss += F.binary_cross_entropy_with_logits(
                preds, 
                binary_targets, 
                pos_weight=pos_weight
            )
            
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Checkpoint loading
checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach6.pt")
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    
    print(f"Resuming training from epoch {start_epoch + 1}")

    if start_epoch >= num_epochs:
        print(f"Model already trained for {start_epoch} epochs, which is >= num_epochs ({num_epochs}).")
        print("Exiting to prevent overwriting or confusion. Increase num_epochs or delete checkpoint to retrain.")
        exit(0)
else:
    print("No checkpoint found. Starting training from scratch.")

print("Starting Training!")
print("=" * 60)

last_epoch = start_epoch
for epoch in range(start_epoch, num_epochs):
    model.train()
    
    total_loss = 0
    num_batches = 0
    
    # Iterate over batches using LinkNeighborLoader
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        
        # Apply edge dropout to the sampled subgraph
        # Note: LinkNeighborLoader returns a subgraph. We can apply dropout here.
        # However, PyG loaders usually return a data object where edge_index is the sampled subgraph.
        # We can't easily 'restore' edges on a subgraph that was constructed by the loader.
        # Instead, we can apply dropout mask to the edge_index in the batch before passing to model.
        # Or we can rely on the fact that sampling itself is a form of regularization (Edge Dropout is less critical).
        # If we really want explicit edge dropout:
        if dropout_edge_prob > 0:
             # Apply mask to 'interact' edges in the batch
             if ('user', 'interact', 'item') in batch.edge_index_dict:
                 edge_index = batch['user', 'interact', 'item'].edge_index
                 mask = torch.rand(edge_index.size(1), device=device) > dropout_edge_prob
                 batch['user', 'interact', 'item'].edge_index = edge_index[:, mask]
                 
             if ('item', 'interact_by', 'user') in batch.edge_index_dict:
                 rev_edge_index = batch['item', 'interact_by', 'user'].edge_index
                 rev_mask = torch.rand(rev_edge_index.size(1), device=device) > dropout_edge_prob
                 batch['item', 'interact_by', 'user'].edge_index = rev_edge_index[:, rev_mask]

        # Get batch labels
        batch_labels = batch['user', 'supervision', 'item'].edge_label
        
        # Get batch user and item indices for the supervision edges
        # In the sampled batch, nodes are re-indexed. 
        # edge_label_index contains the indices of the supervision edges in the *sampled* subgraph.
        batch_edge_label_index = batch['user', 'supervision', 'item'].edge_label_index
        batch_user_indices = batch_edge_label_index[0]
        batch_item_indices = batch_edge_label_index[1]
        
        # Forward pass
        optimizer.zero_grad()
        # Pass the sampled subgraph (batch) to the model
        logits = model(batch, batch_user_indices, batch_item_indices) 
        
        loss = corn_loss(logits, batch_labels, task_weights, num_classes=4)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")
    last_epoch = epoch + 1
    
    # Save checkpoint periodically
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(models_dir, "hetero_sage_model_approach6.pt")
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
save_path = os.path.join(models_dir, "hetero_sage_model_approach6.pt")
print(f"saving as {last_epoch=}")
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
}, save_path)
print(f"Model checkpoint saved to {save_path}")
