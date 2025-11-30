from torch_geometric.data import HeteroData
import torch
from settings import *
import numpy as np

def load_joint_dataset(task1_file, task2_file):
    data = HeteroData()
    save = [[], []]
    buy = [[], []]
    interact = [[], []]    

    user_id2idx = {}
    item_id2idx = {}
    labels = {}             # (u,i) -> interaction (2 or 3)

    group1_view = {}        # user_id -> [item_id, ...]
    group1_user_ids = set() 

    for filename in [task1_file, task2_file]:
        is_group1 = (filename == task1_file)

        with open(os.path.join(data_dir, filename), "r") as f:
            for line in f:
                user_id, item_id, interaction = line.split("\t")
                interaction = int(interaction)

                if user_id not in user_id2idx:
                    user_id2idx[user_id] = len(user_id2idx)
                if item_id not in item_id2idx:
                    item_id2idx[item_id] = len(item_id2idx)

                u_idx = user_id2idx[user_id]
                i_idx = item_id2idx[item_id]

                if interaction in (2, 3):
                    labels[(user_id, item_id)] = interaction

                    interact[0].append(u_idx)
                    interact[1].append(i_idx)

                    if interaction == 2:
                        save[0].append(u_idx); save[1].append(i_idx)
                    elif interaction == 3:
                        buy[0].append(u_idx);  buy[1].append(i_idx)

                if is_group1 and interaction == 1:
                    group1_view.setdefault(user_id, []).append(item_id)

                if is_group1:
                    group1_user_ids.add(user_id)

    data['user', 'save', 'item'].edge_index      = torch.tensor(save,     dtype=torch.long)
    data['user', 'buy', 'item'].edge_index       = torch.tensor(buy,      dtype=torch.long)
    data['user', 'interact', 'item'].edge_index  = torch.tensor(interact, dtype=torch.long)

    data['item', 'saved_by', 'user'].edge_index   = torch.tensor([save[1], save[0]], dtype=torch.long)
    data['item', 'bought_by', 'user'].edge_index  = torch.tensor([buy[1], buy[0]],   dtype=torch.long)
    data['item', 'interact_by', 'user'].edge_index= torch.tensor([interact[1], interact[0]], dtype=torch.long)

    num_users = len(user_id2idx)
    num_items = len(item_id2idx)
    data['user'].num_nodes = num_users
    data['item'].num_nodes = num_items

    user_deg = torch.zeros(num_users)
    item_deg = torch.zeros(num_items)
    for u, i in zip(interact[0], interact[1]):
        user_deg[u] += 1
        item_deg[i] += 1
    data['user'].deg = torch.log1p(user_deg)
    data['item'].deg = torch.log1p(item_deg)

    user_save_deg = torch.zeros(num_users)
    user_buy_deg  = torch.zeros(num_users)
    item_save_deg = torch.zeros(num_items)
    item_buy_deg  = torch.zeros(num_items)

    for u, i in zip(save[0], save[1]):
        user_save_deg[u] += 1
        item_save_deg[i] += 1
    for u, i in zip(buy[0], buy[1]):
        user_buy_deg[u] += 1
        item_buy_deg[i] += 1

    data['user'].save_deg = torch.log1p(user_save_deg)
    data['user'].buy_deg  = torch.log1p(user_buy_deg)
    data['item'].save_deg = torch.log1p(item_save_deg)
    data['item'].buy_deg  = torch.log1p(item_buy_deg)

    group1_mask = torch.zeros(num_users, dtype=torch.bool)
    for uid in group1_user_ids:
        group1_mask[user_id2idx[uid]] = True
    data['user'].is_group1 = group1_mask   

    return data, user_id2idx, item_id2idx, labels, group1_view


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

