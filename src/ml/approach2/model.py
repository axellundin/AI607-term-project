from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCN
import torch
import torch.nn.functional as F
from torch import nn

class HeteroGAT(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = HeteroConv({
            ('user', 'view', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
             ('item', 'viewed_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('user', 'view', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
             ('item', 'viewed_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4)
        )
        
    def forward(self, data, user_ids, item_ids):
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        # Extract embeddings for batch
        user_emb = x_dict['user'][user_ids] 
        item_emb = x_dict['item'][item_ids] 
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.decoder(edge_emb)
        
        return logits

class HeteroSAGE(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = HeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4)
        )
        
    # def forward(self, data, edge_label_index):
    #     x_dict = {
    #         'user': self.user_embedding.weight,
    #         'item': self.item_embedding.weight
    #     }
        
    #     x_dict = self.conv1(x_dict, data.edge_index_dict)
    #     x_dict = {key: F.relu(x) for key, x in x_dict.items()}
    #     x_dict = self.conv2(x_dict, data.edge_index_dict)
        
    #     # Extract embeddings for batch
    #     user_emb = x_dict['user'][edge_label_index[0]] 
    #     item_emb = x_dict['item'][edge_label_index[1]] 
        
    #     # Decode
    #     edge_emb = torch.cat([user_emb, item_emb], dim=-1)
    #     logits = self.decoder(edge_emb)
        
    #     return logits

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
        logits = self.decoder(edge_emb)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)

class OldHeteroSAGE(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = HeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'cart', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
             ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='max')
        
        self.conv2 = HeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'cart', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='max')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4)
        )
        
    def forward(self, data, user_ids, item_ids):
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        # Extract embeddings for batch
        user_emb = x_dict['user'][user_ids] 
        item_emb = x_dict['item'][item_ids] 
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.decoder(edge_emb)
        
        return logits
    
    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)

class HeteroGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = HeteroConv({
            ('user', 'view', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
             ('item', 'viewed_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='max')
        
        self.conv2 = HeteroConv({
            ('user', 'view', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'viewed_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GCN((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='sum')

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 4)
        )
        
    def forward(self, data, user_ids, item_ids):
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        
        # Extract embeddings for batch
        user_emb = x_dict['user'][user_ids] 
        item_emb = x_dict['item'][item_ids] 
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.decoder(edge_emb)
        
        return logits