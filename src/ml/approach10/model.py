from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import scatter
import torch
import torch.nn.functional as F
from torch import nn


class LightGCNConv(MessagePassing):
    """
    LightGCN convolution layer: simple mean aggregation without learnable weights.
    No feature transformation, no non-linearities - just mean pooling of neighbors.
    Supports bipartite graphs (different source and target node types).
    """
    def __init__(self):
        super().__init__(aggr='mean', flow='source_to_target')
    
    def forward(self, x, edge_index, size=None):
        """
        Args:
            x: Node features - can be a tuple (x_src, x_dst) for bipartite graphs
               or a single tensor for homogeneous graphs
            edge_index: Edge indices of shape (2, num_edges)
            size: Optional tuple (num_src_nodes, num_dst_nodes) for bipartite graphs
        
        Returns:
            Aggregated node features
        """
        # Handle bipartite graphs: x can be a tuple (x_src, x_dst)
        if isinstance(x, tuple):
            x_src, x_dst = x
            # For bipartite, propagate from source to target
            out = self.propagate(edge_index, x=(x_src, x_dst), size=size)
            return out
        else:
            # Homogeneous graph
            return self.propagate(edge_index, x=x, size=size)
    
    def message(self, x_j):
        # Just pass the neighbor features (no transformation)
        return x_j
    
    def update(self, aggr_out):
        # Return aggregated features directly (no transformation, no non-linearity)
        return aggr_out


class HeteroLightGCN(torch.nn.Module):
    """
    LightGCN-based model for heterogeneous graphs.
    Uses true LightGCN layers:
    - No non-linearities between layers
    - No feature transformation (simple mean aggregation)
    - No learnable weights in propagation (only embeddings are learnable)
    - Combines embeddings from all layers using attention weighting
    """
    
    def __init__(self, num_users, num_items, embedding_dim, hidden_channels, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Separate paths for each interaction type
        self.convs_view = nn.ModuleList()
        self.convs_save = nn.ModuleList()
        self.convs_buy = nn.ModuleList()
        for _ in range(num_layers):
            # Use LightGCN layers: simple mean aggregation without learnable weights
            conv_view = HeteroConv({
                ('user', 'view', 'item'): LightGCNConv(),
                ('item', 'viewed_by', 'user'): LightGCNConv(),
            }, aggr='mean')
            conv_save = HeteroConv({
                ('user', 'save', 'item'): LightGCNConv(),
                ('item', 'saved_by', 'user'): LightGCNConv(),
            }, aggr='mean')
            conv_buy = HeteroConv({
                ('user', 'buy', 'item'): LightGCNConv(),
                ('item', 'bought_by', 'user'): LightGCNConv(),
            }, aggr='mean')
            self.convs_view.append(conv_view)
            self.convs_save.append(conv_save)
            self.convs_buy.append(conv_buy)
        
        # Attention modules for layer weighting (num_layers + 1 layers including initial embedding)
        # Each attention module learns to weight the different layers
        self.attention_user_view = nn.Linear(embedding_dim, num_layers + 1)
        self.attention_user_save = nn.Linear(embedding_dim, num_layers + 1)
        self.attention_user_buy = nn.Linear(embedding_dim, num_layers + 1)
        self.attention_item_view = nn.Linear(embedding_dim, num_layers + 1)
        self.attention_item_save = nn.Linear(embedding_dim, num_layers + 1)
        self.attention_item_buy = nn.Linear(embedding_dim, num_layers + 1)
        
        # Decoder: Input is 6 * embedding_dim
        # Each node type has 3 interaction paths * embedding_dim = 3 * embedding_dim
        # User (3 * embedding_dim) + Item (3 * embedding_dim) = 6 * embedding_dim total
        self.decoder = nn.Sequential(
            nn.Linear(6 * embedding_dim, hidden_channels),  # 6 = 3 interaction types * 2 (user+item)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 3)  
        )
        
    def forward(self, data, user_ids, item_ids):
        # Initial embeddings (layer 0)
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }
        
        # Store all layer embeddings for each interaction type (LightGCN style)
        all_embeddings_view = [x_dict.copy()]
        all_embeddings_save = [x_dict.copy()]
        all_embeddings_buy = [x_dict.copy()]
        
        x_dict_view = x_dict.copy()
        x_dict_save = x_dict.copy()
        x_dict_buy = x_dict.copy()

        # LightGCN propagation: no non-linearities between layers
        for layer in range(self.num_layers):
            x_dict_view = self.convs_view[layer](x_dict_view, data.edge_index_dict)
            x_dict_save = self.convs_save[layer](x_dict_save, data.edge_index_dict)
            x_dict_buy = self.convs_buy[layer](x_dict_buy, data.edge_index_dict)
            
            # Store layer embeddings
            all_embeddings_view.append(x_dict_view.copy())
            all_embeddings_save.append(x_dict_save.copy())
            all_embeddings_buy.append(x_dict_buy.copy())
        
        # Combine embeddings from all layers using attention weighting
        # Stack embeddings: shape (num_layers + 1, num_nodes, embedding_dim)
        user_view_stack = torch.stack([emb['user'] for emb in all_embeddings_view], dim=0)  # (num_layers+1, num_users, embedding_dim)
        user_save_stack = torch.stack([emb['user'] for emb in all_embeddings_save], dim=0)
        user_buy_stack = torch.stack([emb['user'] for emb in all_embeddings_buy], dim=0)
        
        item_view_stack = torch.stack([emb['item'] for emb in all_embeddings_view], dim=0)  # (num_layers+1, num_items, embedding_dim)
        item_save_stack = torch.stack([emb['item'] for emb in all_embeddings_save], dim=0)
        item_buy_stack = torch.stack([emb['item'] for emb in all_embeddings_buy], dim=0)
        
        # Compute attention weights using the initial embedding as query
        # Use mean of all layers as query for attention
        user_query_view = torch.mean(user_view_stack, dim=0)  # (num_users, embedding_dim)
        user_query_save = torch.mean(user_save_stack, dim=0)
        user_query_buy = torch.mean(user_buy_stack, dim=0)
        
        item_query_view = torch.mean(item_view_stack, dim=0)  # (num_items, embedding_dim)
        item_query_save = torch.mean(item_save_stack, dim=0)
        item_query_buy = torch.mean(item_buy_stack, dim=0)
        
        # Compute attention scores
        user_attn_view = self.attention_user_view(user_query_view)  # (num_users, num_layers+1)
        user_attn_save = self.attention_user_save(user_query_save)
        user_attn_buy = self.attention_user_buy(user_query_buy)
        
        item_attn_view = self.attention_item_view(item_query_view)  # (num_items, num_layers+1)
        item_attn_save = self.attention_item_save(item_query_save)
        item_attn_buy = self.attention_item_buy(item_query_buy)
        
        # Apply softmax to get attention weights
        user_attn_weights_view = F.softmax(user_attn_view, dim=-1)  # (num_users, num_layers+1)
        user_attn_weights_save = F.softmax(user_attn_save, dim=-1)
        user_attn_weights_buy = F.softmax(user_attn_buy, dim=-1)
        
        item_attn_weights_view = F.softmax(item_attn_view, dim=-1)  # (num_items, num_layers+1)
        item_attn_weights_save = F.softmax(item_attn_save, dim=-1)
        item_attn_weights_buy = F.softmax(item_attn_buy, dim=-1)
        
        # Weighted combination: (num_layers+1, num_nodes, embedding_dim) * (num_nodes, num_layers+1) -> (num_nodes, embedding_dim)
        # einsum: sum over layers l: stack[l, n, d] * weights[n, l] -> result[n, d]
        final_user_view = torch.einsum('lnd,nl->nd', user_view_stack, user_attn_weights_view)
        final_user_save = torch.einsum('lnd,nl->nd', user_save_stack, user_attn_weights_save)
        final_user_buy = torch.einsum('lnd,nl->nd', user_buy_stack, user_attn_weights_buy)
        
        final_item_view = torch.einsum('lnd,nl->nd', item_view_stack, item_attn_weights_view)
        final_item_save = torch.einsum('lnd,nl->nd', item_save_stack, item_attn_weights_save)
        final_item_buy = torch.einsum('lnd,nl->nd', item_buy_stack, item_attn_weights_buy)
        
        # Concatenate embeddings from different interaction types
        final_user_emb = torch.cat([final_user_view, final_user_save, final_user_buy], dim=-1)
        final_item_emb = torch.cat([final_item_view, final_item_save, final_item_buy], dim=-1)
        
        # Extract embeddings for the batch
        user_emb = final_user_emb[user_ids]
        item_emb = final_item_emb[item_ids]
        
        # Decode
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        base_logits = self.decoder(edge_emb)
        
        # Ordinal regression: enforce monotonicity
        logit1 = base_logits[:, 0]
        logit2 = logit1 - F.softplus(base_logits[:, 1])
        logit3 = logit2 - F.softplus(base_logits[:, 2])
        
        logits = torch.stack([logit1, logit2, logit3], dim=1)
        
        return logits
    
    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)

