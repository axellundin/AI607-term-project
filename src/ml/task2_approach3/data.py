from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np

def load_dataset(filename):
    data = HeteroData()
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

            if interaction == 2:
                save[0].append(user_idx)
                save[1].append(item_idx)
            elif interaction == 3:
                buy[0].append(user_idx)
                buy[1].append(item_idx)

    data['user', 'save', 'item'].edge_index = torch.tensor(save, dtype=torch.long)
    data['user', 'buy', 'item'].edge_index = torch.tensor(buy, dtype=torch.long)
    data['user', 'interact', 'item'].edge_index = torch.tensor(interact, dtype=torch.long)

    data['item', 'saved_by', 'user'].edge_index = torch.tensor([save[1], save[0]], dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index = torch.tensor([buy[1], buy[0]], dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index = torch.tensor([interact[1], interact[0]], dtype=torch.long)
    
    # IMPORTANT: Set the number of nodes for each node type\

    num_users = len(user_id2idx)
    num_items = len(item_id2idx)
    data['user'].num_nodes = num_users
    data['item'].num_nodes = num_items

    user_deg = torch.zeros(num_users)
    item_deg = torch.zeros(num_items)

    # interact 기준으로 degree 계산
    for u, i in zip(interact[0], interact[1]):
        user_deg[u] += 1
        item_deg[i] += 1

    # log scale normalization
    data['user'].deg = torch.log1p(user_deg)
    data['item'].deg = torch.log1p(item_deg)


    user_save_deg = torch.zeros(num_users)
    user_buy_deg  = torch.zeros(num_users)
    item_save_deg = torch.zeros(num_items)
    item_buy_deg  = torch.zeros(num_items)

    # save / buy 리스트는 이미 만들어져 있음
    for u, i in zip(save[0], save[1]):
        user_save_deg[u] += 1
        item_save_deg[i] += 1

    for u, i in zip(buy[0], buy[1]):
        user_buy_deg[u] += 1
        item_buy_deg[i] += 1

    # log scale
    data['user'].save_deg = torch.log1p(user_save_deg)
    data['user'].buy_deg  = torch.log1p(user_buy_deg)
    data['item'].save_deg = torch.log1p(item_save_deg)
    data['item'].buy_deg  = torch.log1p(item_buy_deg)


    
    return data, user_id2idx, item_id2idx, labels

def get_negative_samples(positive_interactions, user2idx, item2idx, N):
    negative_labels = {}
    num_negative_samples = 0
    num_users = len(user2idx)
    num_items = len(item2idx)
    user_id_lst = list(user2idx.keys())
    item_id_lst = list(item2idx.keys())
    
    # Optimize negative sampling for speed
    while num_negative_samples < N: 
        # Sample in batch for efficiency
        batch_size = N - num_negative_samples
        # Simple random sampling might be slow if collisions are frequent, but for now keep logic simple
        # user_id = user_id_lst[np.random.choice(num_users)] 
        # This np.random.choice is slow inside loop.
        
        # Let's do it slightly better:
        u_indices = np.random.randint(0, num_users, size=batch_size)
        i_indices = np.random.randint(0, num_items, size=batch_size)
        
        for u_idx, i_idx in zip(u_indices, i_indices):
            if num_negative_samples >= N:
                break
            
            user_id = user_id_lst[u_idx]
            item_id = item_id_lst[i_idx]
            
            if (user_id, item_id) not in positive_interactions and (user_id, item_id) not in negative_labels:
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

