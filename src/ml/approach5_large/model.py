from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from torch import nn
import os

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

        # Deeper and larger decoder
        # Option 1: 3 layers with more width (simpler, ~66K params)
        # 256 -> 256 -> 128 -> 3
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels * 2),  # Wider first layer: 256 -> 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, hidden_channels),      # Middle layer: 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 3)  # Output layer: 128 -> 3
        )
        
        # Option 2: 4 layers (more capacity, ~98K params) - uncomment if 3 layers isn't enough
        # self.decoder = nn.Sequential(
        #     nn.Linear(2 * hidden_channels, hidden_channels * 2),  # 256 -> 256
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_channels * 2, hidden_channels),      # 256 -> 128
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_channels, hidden_channels // 2),    # 128 -> 64
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(hidden_channels // 2, 3)  # 64 -> 3
        # )
        
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
        base_logits = self.decoder(edge_emb) 
        
        logit1 = base_logits[:, 0]
        logit2 = logit1 - F.softplus(base_logits[:, 1]) 
        logit3 = logit2 - F.softplus(base_logits[:, 2]) 
        
        logits = torch.stack([logit1, logit2, logit3], dim=1)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        Load weights from a pretrained approach5 model, excluding the decoder.
        This allows initializing the embeddings and GNN layers from a trained model
        while keeping the new larger decoder randomly initialized.
        
        Args:
            checkpoint_path: Path to the approach5 checkpoint file
            device: Device to load the checkpoint on
            strict: If True, requires exact match of all parameter names
        
        Returns:
            dict: Information about loaded and skipped parameters
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        pretrained_state_dict = checkpoint['model_state_dict']
        
        # Filter out decoder parameters and build a filtered state dict
        filtered_state_dict = {}
        loaded_params = []
        skipped_params = []
        
        for name, param in pretrained_state_dict.items():
            # Skip decoder parameters (they have different shapes in the larger model)
            if 'decoder' in name:
                skipped_params.append(name)
                continue
            
            # Add non-decoder parameters to filtered dict
            filtered_state_dict[name] = param
            loaded_params.append(name)
        
        # Load the filtered state dict (this will skip decoder and handle uninitialized params)
        # Use strict=False to allow missing/uninitialized parameters
        result = self.load_state_dict(filtered_state_dict, strict=False)
        if result is not None:
            missing_keys = result.missing_keys
            unexpected_keys = result.unexpected_keys
        else:
            missing_keys = []
            unexpected_keys = []
        
        # Update skipped_params with any missing keys that weren't decoder-related
        for key in missing_keys:
            if 'decoder' not in key and key not in skipped_params:
                skipped_params.append(key)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PRETRAINED WEIGHT LOADING SUMMARY")
        print(f"{'='*60}")
        print(f"Loaded {len(loaded_params)} parameters from pretrained model")
        print(f"Skipped {len(skipped_params)} parameters (decoder or not found)")
        if missing_keys:
            print(f"\nMissing keys (not in current model): {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        if unexpected_keys:
            print(f"\nUnexpected keys (not in pretrained model): {len(unexpected_keys)}")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        print(f"{'='*60}\n")
        
        return {
            'loaded': loaded_params,
            'skipped': skipped_params,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys
        }

