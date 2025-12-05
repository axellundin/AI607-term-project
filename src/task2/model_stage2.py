from torch_geometric.nn import HeteroConv, SAGEConv
import torch
import torch.nn.functional as F
from torch import nn

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

        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1) 
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
    
    def load_user_embeddings(self, checkpoint_path, device='cpu'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        if 'user_embedding.weight' in state_dict:
            self.user_embedding.weight.data = state_dict['user_embedding.weight'].clone()
            print(f"Loaded user embeddings from checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: 'user_embedding.weight' not found in checkpoint")
        
        # Freeze user embeddings
        self.user_embedding.requires_grad_(False)
        print("User embeddings frozen (requires_grad=False)")
        
        if 'item_embedding.weight' in state_dict:
            self.item_embedding.weight.data = state_dict['item_embedding.weight'].clone()
            print(f"Loaded item embeddings from checkpoint (will remain trainable)")
        
        return checkpoint
    
    def get_model_embedding(self, data):
        self.eval()
        with torch.no_grad():
            x_dict = {
                'user': self.user_embedding.weight,
                'item': self.item_embedding.weight
            }
            x_dict = self.conv1(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, data.edge_index_dict)

            user_emb = x_dict['user'].detach().cpu()
            item_emb = x_dict['item'].detach().cpu()
        return user_emb, item_emb



