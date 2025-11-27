import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCNConv, HANConv
from typing import Dict, List, Optional

class BaseHeteroGNN(nn.Module):
    """
    Base class for heterogeneous GNN models.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 4,
                 conv_type: str = 'SAGE'):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.conv_type = conv_type

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Convolution layers as dicts
        self.convs1 = self._get_conv_dict(self.embedding_dim)
        self.convs2 = self._get_conv_dict(self.embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_channels, self.num_classes)
        )

    def _get_conv_dict(self, in_channels):
        """Get the convolution dictionary based on conv_type and in_channels."""
        conv_class = self._get_conv_class()
        return {
            ('user', 'view', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
            ('user', 'save', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
            ('user', 'buy', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
            ('item', 'viewed_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
            ('item', 'saved_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
            ('item', 'bought_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()), # type: ignore
        }

    def _get_conv_class(self):
        """Get the convolution class based on conv_type."""
        if self.conv_type == 'GAT':
            return GATConv
        elif self.conv_type == 'SAGE':
            return SAGEConv
        elif self.conv_type == 'GCN':
            return GCNConv
        elif self.conv_type == 'HeteroGCN':
            return HeteroConv
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

    def _get_conv_kwargs(self) -> Dict:
        """Get additional kwargs for the convolution layer."""
        if self.conv_type == 'GAT':
            return {'add_self_loops': False} 
        elif self.conv_type == 'GCN':
            return {'add_self_loops': False}
        elif self.conv_type == 'SAGE':
            return {'aggr': 'sum'}
        else:
            return {}

    def forward(self, data, user_ids, item_ids):
        # Initial embeddings
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }

        # First convolution layer
        x_dict = self.hetero_conv(x_dict, self.convs1, data)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # Second convolution layer
        x_dict = self.hetero_conv(x_dict, self.convs2, data)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # Get embeddings for batch
        batch_user_emb = x_dict['user'][user_ids]
        batch_item_emb = x_dict['item'][item_ids]
        edge_emb = torch.cat([batch_user_emb, batch_item_emb], dim=-1)
        logits = self.decoder(edge_emb)

        return logits

    def predict(self, data, user_ids, item_ids):
        """Alias for forward, used for prediction."""
        return self.forward(data, user_ids, item_ids)

    def hetero_conv(self, x_dict, convs, data):

        update_dict = {}
        for key in data.keys():
            if isinstance(key, tuple) and len(key) == 3:
                data_obj = data[key]
                if hasattr(data_obj, 'edge_index'):
                    edge_type = key
                    edge_index = data_obj.edge_index
                    src, rel, dst = edge_type
                    conv_key = '__'.join(edge_type)
                    if conv_key in convs:
                        conv = convs[conv_key]
                        out = conv(x_dict[src], edge_index)
                        if dst not in update_dict:
                            update_dict[dst] = out
                        else:
                            update_dict[dst] += out
        for key, out in update_dict.items():
            x_dict[key] = out
        return x_dict


class HeteroGAT(BaseHeteroGNN):
    """Heterogeneous GAT model."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 4):
        super().__init__(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes, conv_type='GAT')


class HeteroSAGE(BaseHeteroGNN):
    """Heterogeneous GraphSAGE model."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 4, aggr: str = 'mean'):
        super().__init__(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes, conv_type='SAGE')
        # Note: aggr is handled in _get_conv_kwargs


class HeteroGCN(BaseHeteroGNN):
    """Heterogeneous GCN model."""
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 4):
        super().__init__(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes, conv_type='GCN')


class HeteroHAN(nn.Module):
    """
    Heterogeneous Graph Attention Network (HAN) model.
    Uses HANConv which requires full graph metadata.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 4, heads: int = 8):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.heads = heads

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # HANConv layers - will be initialized with metadata during forward pass
        self.han_convs = nn.ModuleList()

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

    def initialize_han_layers(self, metadata):
        """
        Initialize HANConv layers with metadata.
        This should be called before loading a saved model.
        """
        if len(self.han_convs) == 0:
            for _ in range(self.num_layers):
                han_conv = HANConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    metadata=metadata
                )
                self.han_convs.append(han_conv)

    def forward(self, data, user_ids, item_ids):
        # Get metadata from the heterogeneous data
        metadata = data.metadata()

        # Initialize HANConv layers if not already done
        if len(self.han_convs) == 0:
            for _ in range(self.num_layers):
                han_conv = HANConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    metadata=metadata
                )
                # Move to the same device as the embeddings
                device = self.user_embedding.weight.device
                han_conv = han_conv.to(device)
                self.han_convs.append(han_conv)
        else:
            # Ensure all HANConv layers are on the correct device
            device = self.user_embedding.weight.device
            for han_conv in self.han_convs:
                han_conv = han_conv.to(device)

        # Initial embeddings
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }

        # HANConv layers
        for han_conv in self.han_convs:
            # Prepare edge_index_dict
            edge_index_dict = {}
            for edge_type, edge_data in data.items():
                if hasattr(edge_data, 'edge_index'):
                    edge_index_dict[edge_type] = edge_data.edge_index

            new_x_dict = han_conv(x_dict, edge_index_dict)
            
            # HANConv may return None for some node types, keep original embeddings in that case
            for node_type in x_dict.keys():
                if node_type in new_x_dict and new_x_dict[node_type] is not None:
                    x_dict[node_type] = new_x_dict[node_type]
                # If None or missing, keep the original embeddings
            
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # Get embeddings for batch
        batch_user_emb = x_dict['user'][user_ids]
        batch_item_emb = x_dict['item'][item_ids]
        edge_emb = torch.cat([batch_user_emb, batch_item_emb], dim=-1)
        logits = self.decoder(edge_emb)

        return logits

    def predict(self, data, user_ids, item_ids):
        """Alias for forward, used for prediction."""
        return self.forward(data, user_ids, item_ids)


class LightGCN(nn.Module):
    """
    LightGCN model for heterogeneous graphs.
    Simplified GCN without non-linearities and feature transformations.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int,
                 num_layers: int = 3, dropout: float = 0.1, num_classes: int = 4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # LightGCN layers (simplified GCN without non-linearities)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = nn.ModuleDict({
                'user_view_item': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
                'user_save_item': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
                'user_buy_item': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
                'item_viewed_by_user': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
                'item_saved_by_user': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
                'item_bought_by_user': GCNConv(embedding_dim, embedding_dim, add_self_loops=False),
            })
            self.convs.append(conv_dict)

        # Decoder (simplified for LightGCN)
        self.decoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, data, user_ids, item_ids):
        # Initial embeddings
        x_dict = {
            'user': self.user_embedding.weight,
            'item': self.item_embedding.weight
        }

        # Store all layer embeddings for final combination
        all_embeddings = [x_dict]

        # LightGCN layers (no non-linearities)
        for conv_dict in self.convs:
            new_x_dict = {}
            for key in data.keys():
                if isinstance(key, tuple) and len(key) == 3:
                    data_obj = data[key]
                    if hasattr(data_obj, 'edge_index'):
                        edge_type = key
                        edge_index = data_obj.edge_index
                        src, rel, dst = edge_type
                        conv_key = f'{src}_{rel}_{dst}'
                        conv = getattr(conv_dict, conv_key, None)
                        if conv is not None:
                            out = conv(x_dict[src], edge_index)
                            if dst not in new_x_dict:
                                new_x_dict[dst] = out
                            else:
                                new_x_dict[dst] += out

            # Ensure all node types are present
            for node_type in ['user', 'item']:
                if node_type not in new_x_dict:
                    new_x_dict[node_type] = x_dict[node_type]

            x_dict = new_x_dict
            all_embeddings.append(x_dict)

        # Combine embeddings from all layers (LightGCN style)
        final_embeddings = {}
        for node_type in ['user', 'item']:
            stacked_emb = torch.stack([emb[node_type] for emb in all_embeddings], dim=0)
            final_embeddings[node_type] = torch.mean(stacked_emb, dim=0)

        # Get embeddings for batch
        batch_user_emb = final_embeddings['user'][user_ids]
        batch_item_emb = final_embeddings['item'][item_ids]
        edge_emb = torch.cat([batch_user_emb, batch_item_emb], dim=-1)
        logits = self.decoder(edge_emb)

        return logits

    def predict(self, data, user_ids, item_ids):
        """Alias for forward, used for prediction."""
        return self.forward(data, user_ids, item_ids)


def create_model(model_name: str, num_users: int, num_items: int, embedding_dim: int,
                 hidden_channels: int, num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, **kwargs) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model ('GAT', 'SAGE', 'GCN', 'HAN', 'LightGCN')
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        hidden_channels: Hidden channels
        num_layers: Number of layers
        dropout: Dropout rate
        num_classes: Number of output classes
        **kwargs: Additional arguments

    Returns:
        Model instance
    """
    if model_name.upper() == 'GAT':
        return HeteroGAT(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes)
    elif model_name.upper() == 'SAGE':
        return HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels,
                          num_layers, dropout, num_classes)
    elif model_name.upper() == 'GCN':
        return HeteroGCN(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes)
    elif model_name.upper() == 'HAN':
        return HeteroHAN(num_users, num_items, embedding_dim, hidden_channels,
                         num_layers, dropout, num_classes)
    elif model_name.upper() == 'LIGHTGCN':
        return LightGCN(num_users, num_items, embedding_dim, num_layers, dropout, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")