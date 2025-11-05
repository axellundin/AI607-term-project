from settings import *
import torch 
from torch import nn
from ml.approach1.model_interactions import GraphSAGEInteractions
from ml.approach1.construct_interaction_graph import construct_interaction_graph
from ml.approach1.feature_selection import compute_pred_feature_matrix
from util.graphs import InteractionGraph 
import numpy as np
from tqdm import tqdm
from util.metrics import compute_MF1

feature_dim = 3 + 3 + 4

G = InteractionGraph(training_data_filename)

device = torch.device("cpu")
model = GraphSAGEInteractions(feature_dim) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

weights = torch.tensor([1, 1, 1, 1]) / 4
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Fix and get data
print("Constructing data graph")
data = construct_interaction_graph(G)
print("Done")

train_idx = torch.randperm(len(data.x))

epochs = 10
batch_size = 100

# Precompute validation features
print("Retrieving validation queries")
queries = []
val_labels = []
with open(os.path.join(data_dir, val_data_filename), 'r') as f:
    for entry in f.readlines():
        user_id, item_id, interaction = entry.split("\t")
        label = np.zeros(4)
        label[int(interaction)] = 1
        queries.append((user_id, item_id))
        val_labels.append(label)
print("Precomputing validation features")
val_features = compute_pred_feature_matrix(G, queries)


print("Starting training!")
for epoch in range(epochs):
    print(f"Epoch {epoch} out of {epochs}.")
    model.train()
    total_loss = 0 
    num_batches = 0

    train_perm = train_idx[torch.randperm(len(train_idx))]

    for i in tqdm(range(0, len(train_perm), batch_size)):
        batch_idx = train_perm[i:i+batch_size]
        batch_features = data.x[batch_idx,:]
        batch_labels = data.y[batch_idx]

        # Randomly roll-back features 
        rollback_idx = np.random.choice(batch_size, size=int(batch_size/2))
        
        for idx in rollback_idx:
            curr_interaction = batch_features[idx,-4:]
            assert len(curr_interaction) == 4, "Wrong indexing!!"
            if curr_interaction[0] == 0: 
                curr_interaction[:3] = curr_interaction[1:]
                curr_interaction[-1] = 0
            batch_features[idx,-4:] = curr_interaction

        optimizer.zero_grad()
        logits = model(batch_features, data.edge_index)
        loss = criterion(logits, batch_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    val_acc, macro_f1 = compute_MF1(model, data.edge_index, val_features, val_labels)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch < 5:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val Macro-F1: {macro_f1:.4f}")

torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_f1': macro_f1,
        'class_weights': weights.cpu(),
        'hidden_dim': 64,
        'num_layers': 2,
    }, os.path.join(models_dir, "model1"))