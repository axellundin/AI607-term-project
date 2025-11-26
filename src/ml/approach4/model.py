from torch_geometric.nn import GATConv, SAGEConv, GCN
import torch
import torch.nn.functional as F
from torch import nn

class MPSHeteroConv(torch.nn.Module):
    def __init__(self, convs, aggr='sum', out_channels=None):
        super().__init__()
        self.convs = nn.ModuleDict()
        # Convert tuple keys to string keys for ModuleDict compatibility if necessary
        # But ModuleDict keys must be strings.
        # PyG's HeteroConv takes a dict where keys are tuples.
        # We need to store mapping.
        self.keys_mapping = {}
        for key, conv in convs.items():
            # key is ('user', 'view', 'item')
            str_key = '__'.join(key)
            self.convs[str_key] = conv
            self.keys_mapping[str_key] = key
            
        self.aggr = aggr
        
        if aggr == 'attention':
            if out_channels is None:
                raise ValueError("out_channels must be provided for attention aggregation")
            self.att_proj = nn.Linear(out_channels, out_channels)
            self.att_vec = nn.Parameter(torch.randn(out_channels))
            nn.init.xavier_uniform_(self.att_proj.weight)
            nn.init.zeros_(self.att_proj.bias)
            nn.init.xavier_uniform_(self.att_vec.unsqueeze(0))
            
    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        # Group results by destination node type
        results_dict = {} 
        
        for str_key, conv in self.convs.items():
            edge_type = self.keys_mapping[str_key]
            src, rel, dst = edge_type
            
            # Handle edge_index lookup
            if edge_type in edge_index_dict:
                edge_index = edge_index_dict[edge_type]
            else:
                continue
                
            # Check for source features
            if src not in x_dict:
                continue
            x_src = x_dict[src]
            
            # Check for dest features for bipartite convs
            if dst in x_dict:
                x_dst = x_dict[dst]
                # Most PyG convs handle tuple (x_src, x_dst)
                res = conv((x_src, x_dst), edge_index)
            else:
                res = conv(x_src, edge_index)
                
            if dst not in results_dict:
                results_dict[dst] = []
            results_dict[dst].append(res)
            
        # Aggregation
        for dst, results in results_dict.items():
            if not results:
                continue
                
            if len(results) == 1:
                out_dict[dst] = results[0]
                continue

            stacked = torch.stack(results, dim=0) # [R, N, D]
            
            if self.aggr == 'sum':
                out_dict[dst] = stacked.sum(dim=0)
            elif self.aggr == 'mean':
                 out_dict[dst] = stacked.mean(dim=0)
            elif self.aggr == 'max':
                 out_dict[dst] = stacked.max(dim=0)[0]
            elif self.aggr == 'attention':
                # Semantic Attention
                # W h + b -> [R, N, D]
                z = torch.tanh(self.att_proj(stacked))
                # z * q -> [R, N] (sum over D)
                scores = (z * self.att_vec).sum(dim=-1) 
                
                # Softmax over relations (dim 0)
                alpha = F.softmax(scores, dim=0) # [R, N]
                
                # Weighted sum
                # alpha: [R, N] -> [R, N, 1]
                # stacked: [R, N, D]
                out = (stacked * alpha.unsqueeze(-1)).sum(dim=0)
                out_dict[dst] = out
                
        return out_dict

class HeteroGAT(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim) 
        
        self.conv1 = MPSHeteroConv({
            ('user', 'view', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
             ('item', 'viewed_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='attention', out_channels=hidden_channels)
        
        self.conv2 = MPSHeteroConv({
            ('user', 'view', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'save', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('user', 'buy', 'item'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
             ('item', 'viewed_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'saved_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
            ('item', 'bought_by', 'user'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
        }, aggr='attention', out_channels=hidden_channels)

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
        
        self.conv1 = MPSHeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='attention', out_channels=hidden_channels)
        
        self.conv2 = MPSHeteroConv({
            ('user', 'view', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'save', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('user', 'buy', 'item'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'viewed_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'saved_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('item', 'bought_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='attention', out_channels=hidden_channels)

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
        
        # Extract embeddings
        user_emb = x_dict['user'][user_ids]  
        item_emb = x_dict['item'][item_ids]  
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.decoder(edge_emb)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)
