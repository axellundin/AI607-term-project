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

        # 6-layer decoder with skip connections
        # Architecture: 512 -> 512 -> 384 -> 256 -> 192 -> 128 -> 3
        # Skip connections from layers 1->3, 2->4, 3->5
        input_dim = 2 * hidden_channels  # 512
        
        self.decoder_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_channels * 2),      # Layer 1: 512 -> 512
            nn.Linear(hidden_channels * 2, int(hidden_channels * 1.5)),  # Layer 2: 512 -> 384
            nn.Linear(int(hidden_channels * 1.5), hidden_channels),     # Layer 3: 384 -> 256
            nn.Linear(hidden_channels, int(hidden_channels * 0.75)),    # Layer 4: 256 -> 192
            nn.Linear(int(hidden_channels * 0.75), hidden_channels // 2), # Layer 5: 192 -> 128
            nn.Linear(hidden_channels // 2, 3)              # Layer 6: 128 -> 3
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * 2),              
            nn.BatchNorm1d(int(hidden_channels * 1.5)),       
            nn.BatchNorm1d(hidden_channels),                  
            nn.BatchNorm1d(int(hidden_channels * 0.75)),      
            nn.BatchNorm1d(hidden_channels // 2),             
        ])
        
        # Projection layers for skip connections to match dimensions
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_channels * 2, hidden_channels),           # Project layer 1 (512) -> layer 3 (256)
            nn.Linear(int(hidden_channels * 1.5), int(hidden_channels * 0.75)),  # Project layer 2 (384) -> layer 4 (192)
            nn.Linear(hidden_channels, hidden_channels // 2),         # Project layer 3 (256) -> layer 5 (128)
        ])
        
        self.dropout = nn.Dropout(0.3)
        
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
        
        edge_emb = torch.cat([user_emb, item_emb], dim=-1)
        x = edge_emb

        # ----- Layer 1 -----
        x1 = self.decoder_layers[0](x)
        x1 = self.bns[0](x1)          
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # ----- Layer 2 -----
        x2 = self.decoder_layers[1](x1)
        x2 = self.bns[1](x2)          
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        # ----- Layer 3 (with skip from layer 1) -----
        x3 = self.decoder_layers[2](x2)
        x3 = self.bns[2](x3)         
        x1_proj = self.skip_projections[0](x1)
        x3 = x3 + x1_proj
        x3 = F.relu(x3)
        x3 = self.dropout(x3)

        # ----- Layer 4 (with skip from layer 2) -----
        x4 = self.decoder_layers[3](x3)
        x4 = self.bns[3](x4)          
        x2_proj = self.skip_projections[1](x2)
        x4 = x4 + x2_proj
        x4 = F.relu(x4)
        x4 = self.dropout(x4)

        # ----- Layer 5 (with skip from layer 3) -----
        x5 = self.decoder_layers[4](x4)
        x5 = self.bns[4](x5)          
        x3_proj = self.skip_projections[2](x3)
        x5 = x5 + x3_proj
        x5 = F.relu(x5)
        x5 = self.dropout(x5)

        # ----- Layer 6 OUTPUT (no BN, no skip) -----
        base_logits = self.decoder_layers[5](x5)
        
        logit1 = base_logits[:, 0]
        logit2 = logit1 - F.softplus(base_logits[:, 1]) 
        logit3 = logit2 - F.softplus(base_logits[:, 2]) 
        
        logits = torch.stack([logit1, logit2, logit3], dim=1)
        
        return logits

    def predict(self, data, user_ids, item_ids):
        return self.forward(data, user_ids, item_ids)
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu', strict=False):
        """
        Load weights from a pretrained approach5_large model, excluding the decoder.
        This allows initializing the embeddings and GNN layers from a trained model
        while keeping the new 6-layer decoder with skip connections randomly initialized.
        
        Args:
            checkpoint_path: Path to the approach5_large checkpoint file
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
            # Skip decoder parameters (they have different shapes in the new model)
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
