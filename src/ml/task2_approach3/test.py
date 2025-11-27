import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Adjust path
sys.path.append(os.getcwd())
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from ml.task2_approach3.model import HeteroSAGE
from ml.task2_approach3.data import load_dataset
from ml.task2_pixie.main import build_gt_dict, evaluate
from settings import data_dir  # Use absolute paths from settings

def build_idx2id(mapping):
    # {"raw_id": idx} -> ["idx": raw_id]
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_batch(model, data, user_indices, num_items, topk, device, user_interacted_dict=None, batch_size=100):
    """
    Batch evaluation: Score all items for users and return topk.
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Process users in batches to avoid memory issues
        num_batches = (len(user_indices) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(user_indices), batch_size), desc="Processing user batches", total=num_batches):
            batch_end = min(batch_start + batch_size, len(user_indices))
            batch_user_indices = user_indices[batch_start:batch_end]
            num_batch_users = len(batch_user_indices)
            
            user_indices_tensor = torch.tensor(batch_user_indices, device=device, dtype=torch.long)
            
            # Expand for all items: [num_batch_users * num_items]
            user_indices_expanded = user_indices_tensor.repeat_interleave(num_items)  
            item_indices_expanded = torch.arange(num_items, device=device, dtype=torch.long).repeat(num_batch_users)  
            
            # Forward pass (returns Dot Product logits)
            logits = model(data, user_indices_expanded, item_indices_expanded)  # [num_batch_users * num_items, 1]
            logits = logits.view(-1) 
            
            # Sigmoid to get probability (0-1), though topk of logits is same as topk of sigmoid
            scores = torch.sigmoid(logits)  
            scores = scores.view(num_batch_users, num_items)  # [num_batch_users, num_items]
            
            # Exclude interacted items
            if user_interacted_dict is not None:
                for i, user_idx in enumerate(batch_user_indices):
                    if user_idx in user_interacted_dict and len(user_interacted_dict[user_idx]) > 0:
                        exclude_items = torch.tensor(list(user_interacted_dict[user_idx]), device=device, dtype=torch.long)
                        scores[i, exclude_items] = -float('inf')
            
            # TopK
            topk_scores, topk_idx = torch.topk(scores, k=min(topk, num_items), dim=1)
            
            # Store results
            for i, user_idx in enumerate(batch_user_indices):
                results[user_idx] = topk_idx[i].cpu().numpy()
    
    return results

def main():
    train_path = os.path.join(data_dir, "task2_train.tsv")
    val_path = os.path.join(data_dir,"task2_val_queries.tsv")
    val_ans_path = os.path.join(data_dir,"task2_val_answers.tsv")
    
    # MODIFIED: Checkpoint path for approach3
    models_dir = "src/results/models_task2_approach3"
    if not os.path.exists(models_dir):
         print(f"Warning: Model directory {models_dir} does not exist.")
         
    checkpoint_path = os.path.join(models_dir, "hetero_sage_model.pt")
    
    topk = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    data, user2idx, item2idx, labels = load_dataset(os.path.basename(train_path))
    num_users = len(user2idx)
    num_items = len(item2idx)

    # Build interaction history to exclude already seen items (if desired, usually standard in recommender eval)
    user_interacted = defaultdict(set)
    if ('user', 'interact', 'item') in data.edge_index_dict:
        edge_index = data['user', 'interact', 'item'].edge_index
        src_users = edge_index[0].numpy()
        dst_items = edge_index[1].numpy()
        for u, i in zip(src_users, dst_items):
            user_interacted[u].add(i)

    data = data.to(device)

    # idx -> raw_id mapping
    idx2item = build_idx2id(item2idx)

    print("Loading validation queries...")
    val = pd.read_csv(
        val_path,
        names=["user"],
        sep="\t",
        dtype={"user" : str}
    )
    val_answers = pd.read_csv(
        val_ans_path,
        names=["user", "item", "interaction"],
        sep="\t",
        dtype={"user" : str, "item" : str}
    )    

    gt_dict = build_gt_dict(val_answers)

    # Load Model
    embedding_dim = 128
    hidden_channels = 64
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("WARNING: checkpoint not found, using randomly initialized model.")

    predictions = {}
    val_users = val["user"].tolist()
    
    # Filter valid users
    valid_user_indices = []
    valid_user_raws = []
    for u_raw in val_users:
        if u_raw in user2idx:
            valid_user_indices.append(user2idx[u_raw])
            valid_user_raws.append(u_raw)
        else:
            predictions[u_raw] = []
    
    # Batch evaluation
    if len(valid_user_indices) > 0:
        batch_size = 100
        print(f"Evaluating {len(valid_user_indices)} users in batches of {batch_size}...")
        batch_results = gnn_recommend_batch(
            model, data,
            user_indices=valid_user_indices,
            num_items=num_items,
            topk=topk,
            device=device,
            user_interacted_dict=user_interacted,
            batch_size=batch_size
        )
        
        # Convert results
        for u_raw, u_idx in zip(valid_user_raws, valid_user_indices):
            topk_item_indices = batch_results[u_idx]
            rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
            predictions[u_raw] = rec_items_raw

    final_score = evaluate(predictions, gt_dict, k=topk)
    print(f"Final validation score (HeteroSAGE Approach 3): {final_score:.6f}")

if __name__ == "__main__":
    main()

