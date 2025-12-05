from task2.model_stage1 import HeteroSAGE
from task2.data_stage1 import load_combined_dataset, load_validation_dataset, get_negative_samples
import torch.nn.functional as F
from settings import *
from tqdm import tqdm
import argparse
import torch 
import os

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
dropout_edge_prob = args.dropout_edge_prob 

# Load combined dataset from both task1 and task2
print("Loading combined dataset from task1 and task2...")
data, user2idx, item2idx, labels, all_interactions = load_combined_dataset(
    task1_filename="task1_train.tsv",
    task2_filename="task2_train.tsv"
)
print(f"Loaded {len(user2idx)} users (task1 + task2), {len(item2idx)} items")
print(f"Training labels (save/buy only): {len(labels)}")

# Add negative samples 
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

def loss_fn(logits, targets):
    probs = torch.sigmoid(logits)
    
    bce_save = F.binary_cross_entropy(probs[:, 0], targets[:, 0].float())
    bce_buy = F.binary_cross_entropy(probs[:, 1], targets[:, 1].float())
    
    loss = bce_save + bce_buy
    return loss

def encode_labels(labels):
    ordinal_labels = torch.zeros(len(labels), 2, dtype=torch.long)
    for i, label in enumerate(labels):
        if label == 2:  
            ordinal_labels[i] = torch.tensor([1, 0])
        elif label == 3:  
            ordinal_labels[i] = torch.tensor([1, 1])
    return ordinal_labels

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 

# Checkpoint loading
checkpoint_path = os.path.join(models_dir, args.model_name)
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

    perm = torch.randperm(len(train_pairs))
    original_edge_index = data['user', 'interact', 'item'].edge_index
    original_rev_edge_index = data['item', 'interact_by', 'user'].edge_index
    
    # Apply mask (dropout)
    mask = torch.rand(original_edge_index.size(1), device=device) > dropout_edge_prob
    data['user', 'interact', 'item'].edge_index = original_edge_index[:, mask]
    rev_mask = torch.rand(original_rev_edge_index.size(1), device=device) > dropout_edge_prob
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index[:, rev_mask]
    
    total_loss = 0
    num_batches = 0
    
    for i in tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_indices = perm[i:i+batch_size]
        
        batch_pairs = [train_pairs[idx] for idx in batch_indices]
        batch_labels_raw = [train_labels[idx] for idx in batch_indices]
        
        batch_user_ids = torch.tensor([user2idx[user_id] for user_id, _ in batch_pairs], device=device)
        batch_item_ids = torch.tensor([item2idx[item_id] for _, item_id in batch_pairs], device=device)
        
        batch_labels_ordinal = encode_labels(batch_labels_raw).to(device)
        
        optimizer.zero_grad()
        logits = model(data, batch_user_ids, batch_item_ids) 
        loss = loss_fn(logits, batch_labels_ordinal)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    data['user', 'interact', 'item'].edge_index = original_edge_index
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index

    avg_loss = total_loss / num_batches
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")
    last_epoch = epoch + 1
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

