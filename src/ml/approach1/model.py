import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, user_features_dim, item_features_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.user_proj = nn.Linear(user_features_dim, hidden_dim)
        self.item_proj = nn.Linear(item_features_dim, hidden_dim)
        
        self.user_convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_dim, aggr='max') for _ in range(num_layers)
        ])
        self.item_convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_dim) for _ in range(num_layers)
        ])
        
        self.user_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.item_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = dropout
        
        # Simpler prediction head
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, user_features, item_features, edge_index, edge_label_index):
        user_x = self.user_proj(user_features)
        item_x = self.item_proj(item_features)
        
        # Message passing
        for i, (user_conv, item_conv, user_bn, item_bn) in enumerate(
            zip(self.user_convs, self.item_convs, self.user_bns, self.item_bns)
        ):
            # Message passing: aggregate item info into users
            user_x_new = user_conv((item_x, user_x), edge_index.flip(0))
            user_x_new = user_bn(user_x_new)
            user_x_new = F.relu(user_x_new)
            user_x_new = F.dropout(user_x_new, p=self.dropout, training=self.training)
            
            # Message passing: aggregate user info into items
            item_x_new = item_conv((user_x, item_x), edge_index)
            item_x_new = item_bn(item_x_new)
            item_x_new = F.relu(item_x_new)
            item_x_new = F.dropout(item_x_new, p=self.dropout, training=self.training)
            
            # Simple update (no residual to save memory)
            user_x = user_x_new
            item_x = item_x_new
        
        # Get embeddings for prediction edges
        user_ids, item_ids = edge_label_index
        u_h = user_x[user_ids]
        i_h = item_x[item_ids]
        x = torch.cat([u_h, i_h], dim=1)
        
        return self.fc(x)