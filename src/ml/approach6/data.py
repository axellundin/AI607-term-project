from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np

def load_combined_dataset(task1_filename="task1_train.tsv", task2_filename="task2_train.tsv"):
    """
    Load both task1 and task2 training data to build a unified graph.
    Returns combined data and user partitions.
    """
    data = HeteroData()
    view = [[], []]
    save = [[], []]
    buy = [[], []]
    interact = [[], []]  # Combined edge type for all interactions

    user_id2idx = {}
    item_id2idx = {}
    labels = {}  # All interactions for training
    
    task1_users = set()
    task2_users = set()

    # Load task1 data
    with open(os.path.join(data_dir, task1_filename), "r") as file:
        for line in file: 
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            
            task1_users.add(user_id)
            labels[(user_id, item_id)] = interaction

            # Mapping from id to index
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            # Add to combined interaction edge type
            interact[0].append(user_idx)
            interact[1].append(item_idx)

            if interaction == 1:
                view[0].append(user_idx)
                view[1].append(item_idx)
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
            
            task2_users.add(user_id)
            labels[(user_id, item_id)] = interaction

            # Mapping from id to index
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            # Add to combined interaction edge type
            interact[0].append(user_idx)
            interact[1].append(item_idx)

            # Task 2 has no views, only save and buy
            if interaction == 2:
                save[0].append(user_idx)
                save[1].append(item_idx)
            elif interaction == 3:
                buy[0].append(user_idx)
                buy[1].append(item_idx)

    data['user', 'view', 'item'].edge_index = torch.tensor(view, dtype=torch.long) if len(view[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'save', 'item'].edge_index = torch.tensor(save, dtype=torch.long) if len(save[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'buy', 'item'].edge_index = torch.tensor(buy, dtype=torch.long) if len(buy[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['user', 'interact', 'item'].edge_index = torch.tensor(interact, dtype=torch.long) if len(interact[0]) > 0 else torch.empty((2, 0), dtype=torch.long)

    data['item', 'viewed_by', 'user'].edge_index = torch.tensor([view[1], view[0]], dtype=torch.long) if len(view[1]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'saved_by', 'user'].edge_index = torch.tensor([save[1], save[0]], dtype=torch.long) if len(save[1]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([buy[1], buy[0]], dtype=torch.long) if len(buy[1]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index = torch.tensor([interact[1], interact[0]], dtype=torch.long) if len(interact[1]) > 0 else torch.empty((2, 0), dtype=torch.long)
    
    # IMPORTANT: Set the number of nodes for each node type
    data['user'].num_nodes = len(user_id2idx)
    data['item'].num_nodes = len(item_id2idx)
    
    return data, user_id2idx, item_id2idx, labels, list(task1_users), list(task2_users)

def get_negative_samples(positive_interactions, user2idx, item2idx, N, task1_users, task2_users, use_task2_neg_sampling=True):
    """
    Generate negative samples.
    If use_task2_neg_sampling is False, only sample users from task1_users.
    """
    negative_labels = {}
    num_negative_samples = 0
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    # Determine user pool
    if use_task2_neg_sampling:
        user_id_lst = list(user2idx.keys())
    else:
        # Only use task 1 users
        # Ensure we only use task1 users that are actually in the map (should be all of them)
        user_id_lst = [u for u in task1_users if u in user2idx]
        
    item_id_lst = list(item2idx.keys())
    
    while num_negative_samples < N: 
        user_id = user_id_lst[np.random.choice(len(user_id_lst))]
        item_id = item_id_lst[np.random.choice(num_items)]
        
        if (user_id, item_id) in positive_interactions:
            continue
        
        # Also check if we already generated this negative sample
        if (user_id, item_id) in negative_labels:
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
