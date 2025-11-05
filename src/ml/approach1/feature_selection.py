from settings import *
from util.graphs import *
import numpy as np
import torch

def safe_divide(numerator, denominator, default=0.0):
    return numerator / denominator if denominator != 0 else default

def compute_pred_feature_matrix(G: InteractionGraph, queries) -> torch.tensor:
    num_predictions = len(queries)

    num_user_features = 3
    num_item_features = 3
    num_joint_features = 4
    num_features = num_user_features + num_item_features + num_joint_features
    feature_matrix = np.zeros((num_predictions, num_features))

    # Compute features for positive samples
    interactions = torch.tensor((num_predictions, 4))
    for idx, (user_id, item_id) in enumerate(tqdm(queries)):
        interaction = 0
        if (user_id, item_id) in G.user_item_to_interaction:
            interaction = G.user_item_to_interaction[(user_id, item_id)]
        feature_matrix[idx,:] = compute_feature(G, user_id, item_id, interaction)
        interactions[idx, interaction] = 1

    return torch.tensor(feature_matrix)

def compute_feature_matrix_with_negative_sampling(G: InteractionGraph) -> torch.tensor:
    num_interactions = len(G.user_item_to_interaction)
    negative_sampling_fraction = 1/4
    negative_samples = int(num_interactions * negative_sampling_fraction)

    num_user_features = 3
    num_item_features = 3
    num_joint_features = 4
    num_features = num_user_features + num_item_features + num_joint_features
    feature_matrix = np.zeros((num_interactions + negative_samples, num_features))

    # Compute features for positive samples
    interactions = torch.zeros((num_interactions + negative_samples, 4))
    for idx, ((user_id, item_id), interaction) in enumerate(tqdm(G.user_item_to_interaction.items())):
        feature_matrix[idx,:] = compute_feature(G, user_id, item_id, interaction)
        interactions[idx, interaction] = 1
    
    # Negative sampling
    samples_collected = 0
    num_users = len(G.user_to_items.keys())
    num_items = len(G.item_to_users.keys())
    negative_interaction_pairs = []
    while samples_collected < negative_samples:
        user_id = str(np.random.choice(num_users) + 1)
        item_id = str(np.random.choice(num_items) + 1)

        if (user_id, item_id) in G.user_item_to_interaction.keys():
            continue

        feature_matrix[num_interactions + samples_collected] = compute_feature(G, user_id, item_id, 0)
        interactions[num_interactions + samples_collected, 0] = 1
        negative_interaction_pairs.append((user_id, item_id))
        samples_collected += 1

    return torch.tensor(feature_matrix), negative_interaction_pairs, interactions

def compute_feature(G:InteractionGraph, 
                    user_id:str, 
                    item_id:str, 
                    interaction:int):
    # Compute behavioral ratios for user
    user_buy_count = len(G.user_to_items_buy[user_id])
    user_save_count = len(G.user_to_items_save[user_id])
    user_view_count = len(G.user_to_items_view[user_id])
    
    buy_turnover = safe_divide(user_buy_count, user_save_count + user_buy_count)
    save_turnover = safe_divide(user_save_count, user_save_count + user_view_count)
    total_num_interactions = len(G.user_to_items[user_id])

    user_feature = np.array([buy_turnover, save_turnover, total_num_interactions])

    # Compute behavioral ratios for item
    item_buy_count = len(G.item_to_users_buy[item_id])
    item_save_count = len(G.item_to_users_save[item_id])
    item_view_count = len(G.item_to_users_view[item_id])
    
    buy_turnover = safe_divide(item_buy_count, item_save_count + item_buy_count)
    save_turnover = safe_divide(item_save_count, item_save_count + item_view_count)
    total_num_interactions = len(G.item_to_users[item_id])

    item_feature = np.array([buy_turnover, save_turnover, total_num_interactions])

    # One hot enc. of interaction type
    interaction_feature = np.zeros(4)
    interaction_feature[interaction] = 1

    return np.hstack((user_feature, item_feature, interaction_feature))
    

