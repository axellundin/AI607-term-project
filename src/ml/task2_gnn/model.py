from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCN
import torch
import torch.nn.functional as F
from torch import nn


class HeteroSAGE(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = HeteroConv({
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels+2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
        

    def forward(self, data, user_ids, item_ids):
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        # Extract embeddings
        user_emb = x_dict['user'][user_ids]  
        item_emb = x_dict['item'][item_ids]  

        user_deg      = data['user'].deg[user_ids].unsqueeze(-1)       # interact degree
        # user_save_deg = data['user'].save_deg[user_ids].unsqueeze(-1) # not good
        # user_buy_deg  = data['user'].buy_deg[user_ids].unsqueeze(-1)

        item_deg      = data['item'].deg[item_ids].unsqueeze(-1)
        # item_save_deg = data['item'].save_deg[item_ids].unsqueeze(-1)
        # item_buy_deg  = data['item'].buy_deg[item_ids].unsqueeze(-1)

        # user_aug = torch.cat([user_emb, user_deg, user_save_deg, user_buy_deg], dim=-1)
        # item_aug = torch.cat([item_emb, item_deg, item_save_deg, item_buy_deg], dim=-1)
        user_aug = torch.cat([user_emb, user_deg], dim=-1)
        item_aug = torch.cat([item_emb, item_deg], dim=-1)

        edge_emb = torch.cat([user_aug, item_aug], dim=-1)
        logits = self.decoder(edge_emb)
                
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)
