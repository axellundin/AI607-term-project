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

        # MLP Decoder removed in favor of Dot Product
        # self.decoder = nn.Sequential(...)
        

    def forward(self, data, user_ids, item_ids):
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.5, training=self.training) for key, x in x_dict.items()} # Added dropout
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        # Extract embeddings
        user_emb = x_dict['user'][user_ids]  
        item_emb = x_dict['item'][item_ids]  

        # Dot Product Decoder
        # Element-wise product then sum across feature dimension
        logits = (user_emb * item_emb).sum(dim=1, keepdim=True)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)

