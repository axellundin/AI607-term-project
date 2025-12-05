import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse

sys.path.append(os.getcwd())
if os.path.join(os.getcwd(), 'src') not in sys.path:
    sys.path.append(os.path.join(os.getcwd(), 'src'))

from util.metrics import evaluate_DCG
from task2.model_stage2 import HeteroSAGE
from task2.data_stage2 import load_combined_dataset
from task2.data_stage2 import build_gt_dict
from settings import *

def build_idx2id(mapping):
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_batch(model, data, user_indices, num_items, topk, device, user_interacted_dict=None, batch_size=100):
    model.eval()
    results = {}
    
    with torch.no_grad():
        num_batches = (len(user_indices) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(user_indices), batch_size), desc="Processing user batches", total=num_batches):
            batch_end = min(batch_start + batch_size, len(user_indices))
            batch_user_indices = user_indices[batch_start:batch_end]
            num_batch_users = len(batch_user_indices)
            
            user_indices_tensor = torch.tensor(batch_user_indices, device=device, dtype=torch.long)
            
            user_indices_expanded = user_indices_tensor.repeat_interleave(num_items)  
            item_indices_expanded = torch.arange(num_items, device=device, dtype=torch.long).repeat(num_batch_users)  
            
            logits = model(data, user_indices_expanded, item_indices_expanded)  
            logits,_= torch.max(logits, dim=-1) 
            logits = logits.squeeze(-1)  
            
            scores = torch.sigmoid(logits)  
            scores = scores.view(num_batch_users, num_items) 
            
            if user_interacted_dict is not None:
                for i, user_idx in enumerate(batch_user_indices):
                    if user_idx in user_interacted_dict and len(user_interacted_dict[user_idx]) > 0:
                        exclude_items = torch.tensor(list(user_interacted_dict[user_idx]), device=device, dtype=torch.long)
                        scores[i, exclude_items] = -float('inf')
            
            topk_scores, topk_idx = torch.topk(scores, k=min(topk, num_items), dim=1)
            
            # Store results
            for i, user_idx in enumerate(batch_user_indices):
                results[user_idx] = topk_idx[i].cpu().numpy()
    
    return results

def save_submission(predictions, users_order, idx2item, out_path, topk=50):
    with open(out_path, "w") as f:
        for u_raw in users_order:
            item_indices = predictions.get(u_raw, [])
            items_raw = [idx2item[int(i)] for i in item_indices[:topk]]
            line = "\t".join([u_raw] + items_raw)
            f.write(line + "\n")
    print(f"Saved submission to {out_path}")

def evaluate_validation_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="task2_stage2.pt")
    args = parser.parse_args()
    # File paths
    train_path = os.path.join(data_dir, "task2_train.tsv")
    val_path = os.path.join(data_dir, "task2_val_queries.tsv")
    val_ans_path = os.path.join(data_dir, "task2_val_answers.tsv")
    checkpoint_path = os.path.join(models_dir, args.input_path)
    topk = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load combined dataset (task1 + task2)
    print("Loading combined dataset (task1 + task2) for graph...")
    data, user2idx, item2idx, labels, all_interactions, task1_users = load_combined_dataset(
        task1_filename="task1_train.tsv",
        task2_filename="task2_train.tsv"
    )
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Loaded {num_users} users, {num_items} items")

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

    idx2item = build_idx2id(item2idx)

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

    gt_dict = build_gt_dict(val_answers)
    print(f"Loaded {len(val)} validation users with ground truth")

    print(f"\nLoading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Please train the model first.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    embedding_dim = checkpoint['hyperparameters']['embedding_dim']
    hidden_channels = checkpoint['hyperparameters']['hidden_channels']
    
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    predictions = {}
    val_users = val["user"].tolist()
    
    valid_user_indices = []
    valid_user_raws = []
    for u_raw in val_users:
        if u_raw in user2idx:
            valid_user_indices.append(user2idx[u_raw])
            valid_user_raws.append(u_raw)
        else:
            predictions[u_raw] = []
    
    print(f"Evaluating {len(valid_user_indices)} valid users (out of {len(val_users)} total)")

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
        
        for u_raw, u_idx in zip(valid_user_raws, valid_user_indices):
            topk_item_indices = batch_results[u_idx]
            rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
            predictions[u_raw] = rec_items_raw

    print("\nEvaluating recommendations...")
    final_score = evaluate_DCG(predictions, gt_dict, k=topk)
    print(f"\n{'='*60}")
    print(f"Final validation score (Normalized DCG-weighted-recall@{topk}): {final_score:.6f}")
    print(f"{'='*60}")

def main():
    """Computes predictions for test set and saves them to a file. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="task2_stage2.pt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    topk = 50

    data, user2idx, item2idx, labels, all_interactions, task1_users = load_combined_dataset(
        task1_filename="task1_train.tsv",
        task2_filename="task2_train.tsv"
    )
    num_items = len(item2idx)

    user_interacted = defaultdict(set)
    if ('user', 'interact', 'item') in data.edge_index_dict:
        edge_index = data['user', 'interact', 'item'].edge_index
        for u, i in zip(edge_index[0].numpy(), edge_index[1].numpy()):
            user_interacted[u].add(i)

    data = data.to(device)
    idx2item = build_idx2id(item2idx)

    test_users = pd.read_csv(os.path.join(data_dir, "task2_test_queries.tsv"), names=["user"], sep="\t", dtype={"user": str})["user"].tolist()

    checkpoint = torch.load(os.path.join(models_dir, args.input_path), map_location=device)
    model = HeteroSAGE(len(user2idx), num_items, checkpoint['hyperparameters']['embedding_dim'], checkpoint['hyperparameters']['hidden_channels']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    predictions = {}
    valid_user_indices = []
    valid_user_raws = []
    for u_raw in test_users:
        if u_raw in user2idx:
            valid_user_indices.append(user2idx[u_raw])
            valid_user_raws.append(u_raw)
        else:
            predictions[u_raw] = []

    if len(valid_user_indices) > 0:
        batch_results = gnn_recommend_batch(model, data, user_indices=valid_user_indices, num_items=num_items, topk=topk, device=device, user_interacted_dict=user_interacted, batch_size=100)
        for u_raw, u_idx in zip(valid_user_raws, valid_user_indices):
            predictions[u_raw] = batch_results[u_idx]

    save_submission(predictions, test_users, idx2item, os.path.join(results_dir, "task2_test_answers.tsv"), topk=topk)

if __name__ == "__main__":
    main()

