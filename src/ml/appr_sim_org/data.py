import os
import torch
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple, Optional

class HeteroGraphDataLoader:
    """
    Generalized data loader for heterogeneous user-item interaction graphs.
    Supports configurable edge types and interaction levels.
    """

    def __init__(self, data_dir: str, interaction_types: Optional[List[str]] = None):
        """
        Args:
            data_dir: Directory containing the data files
            interaction_types: List of interaction types (e.g., ['view', 'save', 'buy'])
                               If None, defaults to ['view', 'save', 'buy']
        """
        self.data_dir = data_dir
        if interaction_types is None:
            self.interaction_types = ['view', 'save', 'buy']
        else:
            self.interaction_types = interaction_types

        # Map interaction levels to indices
        self.interaction_to_idx = {inter: idx + 1 for idx, inter in enumerate(self.interaction_types)}
        self.num_classes = len(self.interaction_types) + 1  # +1 for no interaction
        self.class_count = torch.zeros(self.num_classes)

    def load_dataset(self, filename: str) -> Tuple[HeteroData, Dict, Dict, Dict]:
        """
        Load dataset from TSV file.

        Args:
            filename: Name of the TSV file (without path)

        Returns:
            data: HeteroData object
            user_id2idx: Mapping from user IDs to indices
            item_id2idx: Mapping from item IDs to indices
            labels: Dictionary of (user_id, item_id) -> interaction level
        """
        data = HeteroData()

        # Initialize edge lists for each interaction type
        edge_lists = {inter: [[], []] for inter in self.interaction_types}
        edge_lists['interact'] = [[], []]  # Combined for all interactions

        # Reverse edges
        reverse_edge_lists = {f'{inter}_by': [[], []] for inter in self.interaction_types}
        reverse_edge_lists['interact_by'] = [[], []]

        user_id2idx = {}
        item_id2idx = {}
        labels = {}

        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue  # Skip malformed lines

                user_id, item_id, interaction_str = parts
                interaction = int(interaction_str)

                labels[(user_id, item_id)] = interaction
                self.class_count[interaction] += 1

                # Create indices
                if user_id not in user_id2idx:
                    user_id2idx[user_id] = len(user_id2idx)
                if item_id not in item_id2idx:
                    item_id2idx[item_id] = len(item_id2idx)

                user_idx = user_id2idx[user_id]
                item_idx = item_id2idx[item_id]

                # Add to combined interaction
                edge_lists['interact'][0].append(user_idx)
                edge_lists['interact'][1].append(item_idx)

                # Add to specific interaction type if valid
                if interaction > 0 and interaction <= len(self.interaction_types):
                    inter_type = self.interaction_types[interaction - 1]
                    edge_lists[inter_type][0].append(user_idx)
                    edge_lists[inter_type][1].append(item_idx)

        # Populate reverse edges
        for inter in self.interaction_types + ['interact']:
            if inter == 'view':
                reverse_name = 'viewed_by'
            elif inter == 'save':
                reverse_name = 'saved_by'
            elif inter == 'buy':
                reverse_name = 'bought_by'
            elif inter == 'interact':
                reverse_name = 'interact_by'
            else:
                reverse_name = f'{inter}_by'  # fallback
            
            if reverse_name in reverse_edge_lists and len(reverse_edge_lists[reverse_name][0]) > 0:
                data['item', reverse_name, 'user'].edge_index = torch.tensor(reverse_edge_lists[reverse_name], dtype=torch.long)

        # Set number of nodes
        num_users = len(user_id2idx)
        num_items = len(item_id2idx)
        data['user'].num_nodes = num_users
        data['item'].num_nodes = num_items

        # Create homogeneous edge_index
        # homo_edge_index = []
        # for inter in self.interaction_types + ['interact']:
        #     if inter in edge_lists and len(edge_lists[inter][0]) > 0:
        #         for u, i in zip(edge_lists[inter][0], edge_lists[inter][1]):
        #             homo_edge_index.append([u, num_users + i])
        #             homo_edge_index.append([num_users + i, u])  # bidirectional

        # data.edge_index = torch.tensor(homo_edge_index, dtype=torch.long).t()

        return data, user_id2idx, item_id2idx, labels

    def get_negative_samples(self, positive_interactions: Dict, user2idx: Dict, item2idx: Dict, N: int) -> Dict:
        """
        Generate negative samples (no interaction pairs).

        Args:
            positive_interactions: Dict of positive (user_id, item_id) pairs
            user2idx: User ID to index mapping
            item2idx: Item ID to index mapping
            N: Number of negative samples to generate

        Returns:
            negative_labels: Dict of (user_id, item_id) -> 0
        """
        self.class_count[0] = N
        negative_labels = {}
        num_negative_samples = 0
        num_users = len(user2idx)
        num_items = len(item2idx)
        user_id_lst = list(user2idx.keys())
        item_id_lst = list(item2idx.keys())

        while num_negative_samples < N:
            user_id = user_id_lst[np.random.choice(num_users)]
            item_id = item_id_lst[np.random.choice(num_items)]
            if (user_id, item_id) in positive_interactions:
                continue
            negative_labels[(user_id, item_id)] = 0
            num_negative_samples += 1

        return negative_labels

    def load_validation_dataset(self, filename: str) -> Dict:
        """
        Load validation/test dataset.

        Args:
            filename: Name of the TSV file

        Returns:
            val_data_dict: Dict of (user_id, item_id) -> interaction
        """
        val_data_dict = {}
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                user_id, item_id, interaction = parts
                val_data_dict[(user_id, item_id)] = int(interaction)

        return val_data_dict

    def load_test_queries(self, filename: str) -> List[Tuple[str, str]]:
        """
        Load test queries (user-item pairs without labels).

        Args:
            filename: Name of the TSV file

        Returns:
            queries: List of (user_id, item_id) tuples
        """
        queries = []
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                user_id, item_id = parts
                queries.append((user_id, item_id))

        return queries

    def get_class_weights(self) -> torch.tensor:
        # Make sure that there is no interaction type with zero interactions 
        assert min(self.class_count) > 0, "[DEBUG ERROR] Trying to compute class weights, but not all classes are represented in dataset!"

        num_total = sum(self.class_count)
        weights = num_total / self.class_count
        weights = weights
        
        return weights
