from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np
import pandas as pd 

def load_combined_dataset(task1_filename="task1_train.tsv", task2_filename="task2_train.tsv"):
    data = HeteroData()
    view = [[], []]
    save = [[], []]
    buy = [[], []]
    interact = [[], []]

    user_id2idx = {}
    item_id2idx = {}
    labels = {}
    all_interactions = {}  
    task1_users = set()  
    
    # Load task1 data
    with open(os.path.join(data_dir, task1_filename), "r") as file:
        for line in file: 
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            
            all_interactions[(user_id, item_id)] = interaction
            
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            interact[0].append(user_idx)
            interact[1].append(item_idx)

            task1_users.add(user_id)
            
            if interaction == 1:
                view[0].append(user_idx)
                view[1].append(item_idx)
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
            
            all_interactions[(user_id, item_id)] = interaction
            
            if user_id not in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if item_id not in item_id2idx: 
                item_id2idx[item_id] = len(item_id2idx)

            user_idx = user_id2idx[user_id]
            item_idx = item_id2idx[item_id]

            interact[0].append(user_idx)
            interact[1].append(item_idx)

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

    data['item', 'viewed_by', 'user'].edge_index = torch.tensor([view[1], view[0]], dtype=torch.long) if len(view[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'saved_by', 'user'].edge_index = torch.tensor([save[1], save[0]], dtype=torch.long) if len(save[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([buy[1], buy[0]], dtype=torch.long) if len(buy[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index = torch.tensor([interact[1], interact[0]], dtype=torch.long) if len(interact[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    
    data['user'].num_nodes = len(user_id2idx)
    data['item'].num_nodes = len(item_id2idx)
    
    return data, user_id2idx, item_id2idx, labels, all_interactions, task1_users


def get_negative_samples(all_interactions, user2idx, item2idx, task1_users, N):
    negative_labels = {}
    num_negative_samples = 0
    task1_user_list = [uid for uid in user2idx.keys() if uid in task1_users]
    num_task1_users = len(task1_user_list)
    num_items = len(item2idx)
    item_id_lst = list(item2idx.keys())
    
    while num_negative_samples < N: 
        user_id = task1_user_list[np.random.choice(num_task1_users)]
        item_id = item_id_lst[np.random.choice(num_items)]
        
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
    gt = {}
    for u, group in val_answers.groupby("user"):
        gt[u] = set(group["item"].tolist())
    return gt
