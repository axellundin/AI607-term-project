import torch.nn as nn 
import numpy as np 
from graphs import FullGraph
from datasets import create_dataloader, create_feature_function

class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.0):
        """
        Multi-Layer Perceptron (MLP) with configurable architecture.
        
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes (e.g., [128, 64, 32])
            output_size: Size of output layer
            activation: Activation function ('relu', 'tanh', or 'none')
            dropout: Dropout probability (default: 0.0 for no dropout)
        """
        super(Net, self).__init__()
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        # Build layers
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            # Add activation except for last layer
            if i < len(sizes) - 2:
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier / Glorot uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x) 

def get_predictor_net(feature_size):
    # Output 4 classes: 0 (no interaction), 1 (view), 2 (save), 3 (buy)
    return Net(feature_size, [64, 32], 4, dropout=0.1)

def create_dataset(graph: FullGraph, batch_size=64, shuffle=True):
    """
    Create a training dataloader from a FullGraph.
    
    Args:
        graph: FullGraph object containing user-item interactions
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object for training
    """
    # Create feature extraction function
    feature_fn = create_feature_function(graph)
    
    # Create and return dataloader
    training_data = create_dataloader(
        graph,
        batch_size=batch_size,
        shuffle=shuffle,
        feature_fn=feature_fn
    )
    
    return training_data
