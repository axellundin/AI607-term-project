from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np
import pandas as pd 

def load_combined_dataset(task1_filename="task1_train.tsv", task2_filename="task2_train.tsv"):
    """
    Load both task1 and task2 training data to build a unified graph.
    Graph includes all interaction types (view, save, buy) from both tasks.
    Training labels only include view (1) interactions from task1 users.
    """
    data = HeteroData()
    view = [[], []]
    save = [[], []]
    buy = [[], []]
    interact = [[], []]  # Combined edge type for all interactions

    user_id2idx = {}
    item_id2idx = {}
    labels = {}  # Only view (1) interactions from task1 users for training
    all_interactions = {}  # All interactions (view, save, buy) for negative sampling exclusion
    task1_users = set()  # Track which users belong to task1
    
    # Load task1 data
    with open(os.path.join(data_dir, task1_filename), "r") as file:
        for line in file: 
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            
            # Track all interactions for negative sampling
            all_interactions[(user_id, item_id)] = interaction
            
            # Mapping from id to index
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            # Add to combined interaction edge type (all interactions for graph)
            interact[0].append(user_idx)
            interact[1].append(item_idx)

            # Track task1 users
            task1_users.add(user_id)
            
            # Add to specific edge types for graph construction
            if interaction == 1:
                view[0].append(user_idx)
                view[1].append(item_idx)
                # Only view interactions from task1 users are used as training labels
                labels[(user_id, item_id)] = interaction
            elif interaction == 2:
                save[0].append(user_idx)
                save[1].append(item_idx)
            elif interaction == 3:
                buy[0].append(user_idx)
                buy[1].append(item_idx)
    
    # Load task2 data
    with open(os.path.join(data_dir, task2_filename), "r") as file:
        for line in file: 
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            
            # Track all interactions for negative sampling
            all_interactions[(user_id, item_id)] = interaction
            
            # Mapping from id to index (users from task2 are disjoint from task1)
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            # Add to combined interaction edge type (all interactions for graph)
            interact[0].append(user_idx)
            interact[1].append(item_idx)

            # Task2 only has save (2) and buy (3), no view
            # These are NOT used as training labels (only task1 view interactions are)
            if interaction == 2:
                save[0].append(user_idx)
                save[1].append(item_idx)
            elif interaction == 3:
                buy[0].append(user_idx)
                buy[1].append(item_idx)

    # Build graph with all interaction types
    data['user', 'view', 'item'].edge_index = torch.tensor(view, dtype=torch.long) if len(view[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'save', 'item'].edge_index = torch.tensor(save, dtype=torch.long) if len(save[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'buy', 'item'].edge_index = torch.tensor(buy, dtype=torch.long) if len(buy[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'interact', 'item'].edge_index = torch.tensor(interact, dtype=torch.long) if len(interact[0]) > 0 else torch.empty((2, 0), dtype=torch.long)

    data['item', 'viewed_by', 'user'].edge_index = torch.tensor([view[1], view[0]], dtype=torch.long) if len(view[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'saved_by', 'user'].edge_index = torch.tensor([save[1], save[0]], dtype=torch.long) if len(save[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([buy[1], buy[0]], dtype=torch.long) if len(buy[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index = torch.tensor([interact[1], interact[0]], dtype=torch.long) if len(interact[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    
    # IMPORTANT: Set the number of nodes for each node type
    data['user'].num_nodes = len(user_id2idx)
    data['item'].num_nodes = len(item_id2idx)
    
    return data, user_id2idx, item_id2idx, labels, all_interactions, task1_users


def get_negative_samples(all_interactions, user2idx, item2idx, task1_users, N):
    """
    Generate negative samples (no interaction) for task1 users only.
    Exclude any existing interaction from either task1 or task2.
    
    Args:
        all_interactions: Dict of all (user_id, item_id) -> interaction from both tasks
        user2idx: User ID to index mapping
        item2idx: Item ID to index mapping
        task1_users: Set of task1 user IDs
        N: Number of negative samples to generate
    
    Returns:
        negative_labels: Dict of (user_id, item_id) -> 0 (no interaction)
    """
    negative_labels = {}
    num_negative_samples = 0
    task1_user_list = [uid for uid in user2idx.keys() if uid in task1_users]
    num_task1_users = len(task1_user_list)
    num_items = len(item2idx)
    item_id_lst = list(item2idx.keys())
    
    while num_negative_samples < N: 
        user_id = task1_user_list[np.random.choice(num_task1_users)]
        item_id = item_id_lst[np.random.choice(num_items)]
        
        # Exclude any existing interaction (view, save, or buy) from either task
        if (user_id, item_id) in all_interactions:
            continue
        
        negative_labels[(user_id, item_id)] = 0
        num_negative_samples += 1

    return negative_labels

def load_validation_dataset(filename):
    val_data_dict = {}

    with open(os.path.join(data_dir, filename), "r") as file:
        for line in file: 
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            val_data_dict[(user_id, item_id)] = interaction

    return val_data_dict

def demo():
    dict = load_validation_dataset(val_data_filename)
    print(len(dict))
    data, ud, id, labels, all_interactions, task1_users = load_combined_dataset()

def build_gt_dict(val_answers: pd.DataFrame):
    """
    user -> set(items) 로 ground-truth 뷰 아이템을 만듦
    (val_answers는 interaction=1이니까 그냥 전부 넣으면 됨)
    """
    gt = {}
    for u, group in val_answers.groupby("user"):
        gt[u] = set(group["item"].tolist())
    return gt

def dcg_weights(k: int = 50):
    weights = np.array([1.0 / np.log2(j + 1) for j in range(1, k + 1)], dtype=np.float64)
    return weights

def evaluate(predictions: dict, gt_dict: dict, k: int = 50) -> float:
    """
    predictions: user -> list of item_ids (length <= k)
    gt_dict: user -> set of ground-truth viewed items
    return: final score (mean of normalized scores)
    """
    weights = dcg_weights(k)
    max_score = weights.sum() 
    scores = []

    for u, rec_items in predictions.items():
        gt_items = gt_dict.get(u, set())

        # S(u) = sum_{j} w_j * 1[rec_j in Gu]
        s_u = 0.0
        for j, item in enumerate(rec_items[:k]):  # j = 0..k-1
            if item in gt_items:
                s_u += weights[j]

        # normalized
        s_u_tilde = s_u / max_score if max_score > 0 else 0.0
        scores.append(s_u_tilde)

    if not scores:
        return 0.0
    return float(np.mean(scores))
