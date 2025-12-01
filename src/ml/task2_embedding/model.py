from torch_geometric.nn import HeteroConv, SAGEConv
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
            ('user', 'interact', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'interact_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'interact', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'interact_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 2)  # Output 2 logits: save and buy
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
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        base_logits = self.decoder(edge_emb)  # Shape: (batch_size, 2)
        
        # Enforce monotonicity: logit_save >= logit_buy
        # This ensures p_save >= p_buy after sigmoid, respecting ordinal structure
        logit_save = base_logits[:, 0]
        logit_buy = logit_save - F.softplus(base_logits[:, 1])  # Ensures logit_buy <= logit_save
        
        logits = torch.stack([logit_save, logit_buy], dim=1)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)

