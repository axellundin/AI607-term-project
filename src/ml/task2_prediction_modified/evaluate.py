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

from ml.task2_prediction.model import HeteroSAGE
from ml.task2_prediction.data import load_combined_dataset
from ml.task2_pixie.main import build_gt_dict, evaluate
from settings import data_dir, models_dir

def build_idx2id(mapping):
    """
    Convert ID to index mapping to index to ID mapping.
    {"raw_id": idx} -> {idx: raw_id}
    """
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_batch(model, data, user_indices, num_items, topk, device, user_interacted_dict=None, batch_size=100):
    """
    Batch evaluation: Score all items for users and return topk.
    Excludes items that users have already interacted with (save/buy from training).
    
    Args:
        model: Trained HeteroSAGE model
        data: Graph data for message passing
        user_indices: List of user indices to evaluate
        num_items: Total number of items
        topk: Number of top items to recommend
        device: Device to run on
        user_interacted_dict: Dict mapping user_idx -> set of item_indices to exclude
        batch_size: Number of users to process per batch
    
    Returns:
        results: Dict mapping user_idx -> array of topk item indices
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
            
            # Forward pass (returns binary classifier logits for view prediction)
            logits = model(data, user_indices_expanded, item_indices_expanded)  # [num_batch_users * num_items, 1]
            logits = logits.squeeze(-1)  # [num_batch_users * num_items]
            
            # Sigmoid to get probability (0-1) for view prediction
            scores = torch.sigmoid(logits)  
            scores = scores.view(num_batch_users, num_items)  # [num_batch_users, num_items]
            
            # Exclude interacted items (items user has save/buy interactions with)
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
    """
    Main evaluation function for task2_prediction model.
    Evaluates on task2 validation dataset using normalized DCG-weighted-recall.
    """
    # File paths
    train_path = os.path.join(data_dir, "task2_train.tsv")
    val_path = os.path.join(data_dir, "task2_val_queries.tsv")
    val_ans_path = os.path.join(data_dir, "task2_val_answers.tsv")
    checkpoint_path = os.path.join(models_dir, "hetero_sage_model_task2_prediction.pt")
    topk = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load combined dataset (task1 + task2) for graph construction
    print("Loading combined dataset (task1 + task2) for graph...")
    data, user2idx, item2idx, labels, all_interactions, task1_users = load_combined_dataset(
        task1_filename="task1_train.tsv",
        task2_filename="task2_train.tsv"
    )
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Loaded {num_users} users, {num_items} items")

    # Build interaction history: track items each user has interacted with (save/buy from training)
    # These items should be excluded from recommendations
    print("Building user interaction history...")
    user_interacted = defaultdict(set)
    if ('user', 'interact', 'item') in data.edge_index_dict:
        edge_index = data['user', 'interact', 'item'].edge_index
        src_users = edge_index[0].numpy()
        dst_items = edge_index[1].numpy()
        for u, i in zip(src_users, dst_items):
            user_interacted[u].add(i)
    print(f"Built interaction history for {len(user_interacted)} users")

    data = data.to(device)

    # Build index to raw ID mappings
    idx2item = build_idx2id(item2idx)

    # Load task2 validation queries and answers
    print("Loading task2 validation data...")
    val = pd.read_csv(
        val_path,
        names=["user"],
        sep="\t",
        dtype={"user": str}
    )
    val_answers = pd.read_csv(
        val_ans_path,
        names=["user", "item", "interaction"],
        sep="\t",
        dtype={"user": str, "item": str}
    )
    
    # Build ground truth dictionary (user -> set of viewed items)
    gt_dict = build_gt_dict(val_answers)
    print(f"Loaded {len(val)} validation users with ground truth")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Please train the model first.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    embedding_dim = checkpoint['hyperparameters']['embedding_dim']
    hidden_channels = checkpoint['hyperparameters']['hidden_channels']
    
    # Initialize model
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Filter valid users (users that exist in training set)
    predictions = {}
    val_users = val["user"].tolist()
    
    valid_user_indices = []
    valid_user_raws = []
    for u_raw in val_users:
        if u_raw in user2idx:
            valid_user_indices.append(user2idx[u_raw])
            valid_user_raws.append(u_raw)
        else:
            # User not in training set, return empty recommendations
            predictions[u_raw] = []
    
    print(f"Evaluating {len(valid_user_indices)} valid users (out of {len(val_users)} total)")

    # Batch evaluation for all valid users
    if len(valid_user_indices) > 0:
        batch_size = 100
        print(f"Generating top-{topk} recommendations in batches of {batch_size} users...")
        batch_results = gnn_recommend_batch(
            model, data,
            user_indices=valid_user_indices,
            num_items=num_items,
            topk=topk,
            device=device,
            user_interacted_dict=user_interacted,
            batch_size=batch_size
        )
        
        # Convert results to raw item IDs
        for u_raw, u_idx in zip(valid_user_raws, valid_user_indices):
            topk_item_indices = batch_results[u_idx]
            rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
            predictions[u_raw] = rec_items_raw

    # Evaluate using normalized DCG-weighted-recall
    print("\nEvaluating recommendations...")
    final_score = evaluate(predictions, gt_dict, k=topk)
    print(f"\n{'='*60}")
    print(f"Final validation score (Normalized DCG-weighted-recall@{topk}): {final_score:.6f}")
    print(f"{'='*60}")
    
    return final_score, predictions

if __name__ == "__main__":
    main()

