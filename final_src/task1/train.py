from task1.model import HeteroSAGE
from task1.data import load_dataset, load_validation_dataset, get_negative_samples
import torch.nn.functional as F
from tqdm import tqdm
from settings import *
import torch 
import os

num_epochs = 25
embedding_dim = 256
hidden_channels = 256
batch_size = 16384
learning_rate = 0.005 
dropout_edge_prob = 0.2  

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
model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)

data = data.to(device)


def loss_fn(logits, targets):
    probs = torch.sigmoid(logits)
    
    bce1 = F.binary_cross_entropy(probs[:, 0], targets[:, 0].float())
    bce2 = F.binary_cross_entropy(probs[:, 1], targets[:, 1].float()) 
    bce3 = F.binary_cross_entropy(probs[:, 2], targets[:, 2].float()) 
    
    loss = bce1 + bce2 + bce3
    return loss

def encode_labels(labels):
    ordinal_labels = torch.zeros(len(labels), 3, dtype=torch.long)
    for i, label in enumerate(labels):
        if label == 1:  # view
            ordinal_labels[i] = torch.tensor([1, 0, 0])
        elif label == 2:  # save
            ordinal_labels[i] = torch.tensor([1, 1, 0])
        elif label == 3:  # buy
            ordinal_labels[i] = torch.tensor([1, 1, 1])
    return ordinal_labels

def predict_with_threshold(logits, threshold=0.5):
    probs = torch.sigmoid(logits)  
    b0 = probs[:, 0] > threshold   
    b1 = probs[:, 1] > threshold   
    b2 = probs[:, 2] > threshold   
    pred = b0.int() + b1.int() + b2.int()
    return pred

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 

checkpoint_path = os.path.join(models_dir, "task1.pt")  
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
    
    # Apply mask 
    mask = torch.rand(original_edge_index.size(1), device=device) > dropout_edge_prob
    data['user', 'interact', 'item'].edge_index = original_edge_index[:, mask]
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
        batch_labels_ordinal = encode_labels(batch_labels_raw).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(data, batch_user_ids, batch_item_ids)
        loss = loss_fn(logits, batch_labels_ordinal)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            preds = predict_with_threshold(logits, threshold=0.5)
            batch_labels_tensor = torch.tensor(batch_labels_raw, device=device)
            correct_predictions += (preds == batch_labels_tensor).sum().item()
            total_samples += len(batch_labels_raw)
        
        total_loss += loss.item()
        num_batches += 1
    
    data['user', 'interact', 'item'].edge_index = original_edge_index
    data['item', 'interact_by', 'user'].edge_index = original_rev_edge_index

    avg_loss = total_loss / num_batches
    train_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    last_epoch = epoch + 1
    checkpoint_interval = 1
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(models_dir, "task1.pt")
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
save_path = os.path.join(models_dir, "task1.pt")
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

