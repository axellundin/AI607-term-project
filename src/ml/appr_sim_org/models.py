import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCNConv
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
            ('user', 'view', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
            ('user', 'save', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
            ('user', 'buy', 'item'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
            ('item', 'viewed_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
            ('item', 'saved_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
            ('item', 'bought_by', 'user'): conv_class(in_channels, self.embedding_dim, **self._get_conv_kwargs()),
        }

    def _get_conv_class(self):
        """Get the convolution class based on conv_type."""
        if self.conv_type == 'GAT':
            return GATConv
        elif self.conv_type == 'SAGE':
            return SAGEConv
        elif self.conv_type == 'GCN':
            return GCNConv
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


def create_model(model_name: str, num_users: int, num_items: int, embedding_dim: int,
                 hidden_channels: int, num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, **kwargs) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model ('GAT', 'SAGE', 'GCN')
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")