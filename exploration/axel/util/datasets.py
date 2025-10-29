import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from graphs import FullGraph
from clustering import compute_embedding


class GraphDataset(Dataset):
    """
    Custom dataset for graph-based recommendation data.
    
    Args:
        graph: FullGraph object containing user-item interactions
        feature_fn: Function to extract features for a (user_id, item_id) pair
    """
    def __init__(self, graph, feature_fn=None):
        self.graph = graph
        self.feature_fn = feature_fn
        
        # Convert user_item_to_interaction to lists
        self.user_item_pairs = list(graph.user_item_to_interaction.keys())
        self.interactions = [graph.user_item_to_interaction[pair] for pair in self.user_item_pairs]
        
        print(f"Dataset created with {len(self.user_item_pairs)} samples")
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.user_item_pairs)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            features: Input features as tensor
            label: Interaction label as tensor
        """
        user_id, item_id = self.user_item_pairs[idx]
        interaction = self.interactions[idx]
        
        # Get features
        if self.feature_fn:
            features = self.feature_fn(user_id, item_id, self.graph)
        else:
            features = self._default_features(user_id, item_id)
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        # Training data has labels 1,2,3. We need to map them to 1,2,3 (not 0,1,2)
        # because class 0 is reserved for "no interaction" which we'll need for validation
        label = torch.LongTensor([interaction])  # Keep 1,2,3 as 1,2,3
        
        return features, label
    
    def _default_features(self, user_id, item_id):
        """Default feature extraction."""
        # Example: user degree, item degree, shared neighbors
        user_deg = len(self.graph.user_to_items[user_id])
        item_deg = len(self.graph.item_to_users[item_id])
        
        # Shared items (items liked by both user and item's users)
        user_items = set(self.graph.user_to_items[user_id])
        item_users = self.graph.item_to_users[item_id]
        
        shared_count = 0
        for item_user in item_users:
            shared_count += len(user_items.intersection(set(self.graph.user_to_items[item_user])))
        
        return [user_deg, item_deg, shared_count]


def create_feature_function(graph, cluster_to_user=None, cluster_to_item=None):
    """Create a feature extraction function specific to your task."""
   
    def extract_features(user_id, item_id, graph_obj:FullGraph):
        """Extract features for a user-item pair."""
        if cluster_to_user is None or cluster_to_item is None:
            raise ValueError("cluster_to_user and cluster_to_item must be provided")
        return compute_embedding(
            user_id, 
            item_id, 
            cluster_to_user, 
            cluster_to_item,
            graph_obj.user_to_items,
            graph_obj.item_to_users
        )
    
    return extract_features


def create_dataloader(graph, batch_size=32, shuffle=True, feature_fn=None):
    """
    Create a DataLoader for the graph dataset.
    
    Args:
        graph: FullGraph object
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        feature_fn: Custom feature extraction function
    
    Returns:
        DataLoader object
    """
    dataset = GraphDataset(graph, feature_fn=feature_fn)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Use 0 for debugging, increase for speed
        pin_memory=False  # Set to True if using GPU
    )
    
    return dataloader

