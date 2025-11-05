import torch
import torch_geometric
from torch_geometric.data import Data
from util.graphs import * 
from ml.approach1.feature_selection import * 

def construct_interaction_graph(G:InteractionGraph) -> Data:
    interaction_pairs = list(G.user_item_to_interaction.keys()) 
    features, negative_sample_pairs, labels = compute_feature_matrix_with_negative_sampling(G)
    interaction_pairs += negative_sample_pairs 
    
    idx_lookup = {}
    for idx, (user_id, item_id) in enumerate(interaction_pairs):
        idx_lookup[(user_id, item_id)] = idx

    edge_index_L = []
    edge_index_R = []
    print("Enumerating edges...")
    for idx, (user_id, item_id) in enumerate(tqdm(interaction_pairs)):
        for adj_item_id in G.user_to_items[user_id]:
            edge_index_L.append(idx)
            edge_index_R.append(idx_lookup[(user_id, adj_item_id)])
        for adj_user_id in G.item_to_users[item_id]:
            edge_index_L.append(idx)
            edge_index_R.append(idx_lookup[(adj_user_id, item_id)])

    edge_index = torch.tensor([edge_index_L, edge_index_R])
    del edge_index_L, edge_index_R
    return Data(x=features, edge_index=edge_index, y=labels, is_undirected=True)
                
if __name__=='__main__':
    filename="task1_train.tsv"
    G = InteractionGraph(filename)
    construct_interaction_graph(G)