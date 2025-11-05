import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEInteractions(nn.Module):
    def __init__(self, input_features_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.feature_proj = nn.Linear(input_features_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            SAGEConv((-1, -1), hidden_dim, aggr='max') for _ in range(num_layers)
        ])
        
        self.user_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = dropout
        
        # Simpler prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, features, edge_index):
        x = self.feature_proj(features)
        
        # Message passing
        for i, (user_conv, user_bn) in enumerate(
            zip(self.convs, self.user_bns)
        ):
            # Message passing: aggregate item info into users
            x_new = user_conv(x, edge_index.flip(0))
            x_new = user_bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            x = x_new
            x = x_new

        return self.fc(x)