from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np

def load_dataset(filename):
    data = HeteroData()
    view = [[], []]
    save = [[], []]
    buy = [[], []]
    interact = [[], []]  # Combined edge type for all interactions

    user_id2idx = {}
    item_id2idx = {}
    labels = {}
    with open(os.path.join(data_dir, filename), "r") as file:
        for line in file: 
            # Get data from file
            user_id, item_id, interaction = line.split("\t")
            interaction = int(interaction)
            labels[(user_id, item_id)] = interaction
            # Mapping from id to index
            if not user_id in user_id2idx: 
                user_id2idx[user_id] = len(user_id2idx)
            if not item_id in item_id2idx: 
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

    data['user', 'view', 'item'].edge_index = torch.tensor(view, dtype=torch.long)
    data['user', 'save', 'item'].edge_index = torch.tensor(save, dtype=torch.long)
    data['user', 'buy', 'item'].edge_index = torch.tensor(buy, dtype=torch.long)
    data['user', 'interact', 'item'].edge_index = torch.tensor(interact, dtype=torch.long)

    data['item', 'viewed_by', 'user'].edge_index = torch.tensor([view[1], view[0]], dtype=torch.long)
    data['item', 'saved_by', 'user'].edge_index = torch.tensor([save[1], save[0]], dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([buy[1], buy[0]], dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index = torch.tensor([interact[1], interact[0]], dtype=torch.long)
    
    # IMPORTANT: Set the number of nodes for each node type
    data['user'].num_nodes = len(user_id2idx)
    data['item'].num_nodes = len(item_id2idx)
    
    return data, user_id2idx, item_id2idx, labels

def get_negative_samples(positive_interactions, user2idx, item2idx, N):
    negative_labels = {}
    num_negative_samples = 0
    num_users = len(user2idx)
    num_items = len(item2idx)
    user_id_lst = list(user2idx.keys())
    item_id_lst = list(item2idx.keys())
    while num_negative_samples < N: 
        user_id = user_id_lst[np.random.choice(num_users)]
        item_id = item_id_lst[np.random.choice(num_items)]
        if (user_id, item_id) in positive_interactions.keys():
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
    data, ud, id = load_dataset(training_data_filename)
