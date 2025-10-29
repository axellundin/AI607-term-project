import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from ml import Net, get_predictor_net
from datasets import create_dataloader, create_feature_function
from graphs import FullGraph


class ValDataset(torch.utils.data.Dataset):
    """Dataset for validation queries."""
    def __init__(self, queries_file, graph, feature_fn):
        self.graph = graph
        self.feature_fn = feature_fn
        self.pairs = []
        
        with open(queries_file, 'r') as f:
            for line in f:
                user_id, item_id = line.strip().split('\t')
                self.pairs.append((user_id, item_id))
        
        print(f"Validation dataset created with {len(self.pairs)} queries")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        user_id, item_id = self.pairs[idx]
        
        try:
            features = self.feature_fn(user_id, item_id, self.graph)
        except KeyError:
            # For unseen pairs, return zero embedding
            num_user_clusters = len(self.graph.cluster_to_user.keys())
            num_item_clusters = len(self.graph.cluster_to_item.keys())
            features = np.zeros(num_user_clusters + num_item_clusters)
        
        features = torch.FloatTensor(features)
        return features, (user_id, item_id)


def load_validation_answers(answers_file):
    """Load ground truth labels for validation set."""
    labels = {}
    with open(answers_file, 'r') as f:
        for line in f:
            user_id, item_id, label = line.strip().split('\t')
            labels[(user_id, item_id)] = int(label)
    return labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features = features.to(device)
        # Training labels are 1,2,3 (no 0 in training data)
        # Model has 4 outputs for 0,1,2,3
        labels = labels.squeeze().to(device)  # Keep 1,2,3 as-is
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        num_samples += features.size(0)
    
    return total_loss / num_samples


def evaluate(model, val_loader, val_labels, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    predictions = {}
    
    with torch.no_grad():
        for features, pairs in tqdm(val_loader, desc="Evaluating"):
            features = features.to(device)
            
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Get predictions for batch
            for i, (user_id, item_id) in enumerate(pairs):
                # Model outputs 4 classes: 0,1,2,3
                # Training data has labels 1,2,3 mapped to outputs 1,2,3 during training
                # So model predictions are: 0 (none), 1 (view), 2 (save), 3 (buy)
                pred = predicted[i].item()
                predictions[(user_id, item_id)] = pred
    
    # Compare with ground truth
    for (uid, iid), pred_label in predictions.items():
        if (uid, iid) in val_labels:
            true_label = val_labels[(uid, iid)]
            
            if pred_label == true_label:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, predictions

def train_and_evaluate(
    train_graph_path,
    val_queries_path,
    val_answers_path,
    num_epochs=20,
    batch_size=64,
    learning_rate=0.001,
    device='cpu'
):
    """Main training and evaluation function."""
    
    print("Loading training graph...")
    train_graph = FullGraph(train_graph_path)
    
    print("Creating feature function...")
    feature_fn = create_feature_function(train_graph)
    
    # Create training dataloader
    print("Creating training dataloader...")
    train_loader = create_dataloader(train_graph, batch_size=batch_size, shuffle=True, feature_fn=feature_fn)
    
    # Get feature size by inspecting one sample
    sample_features = feature_fn(
        list(train_graph.user_to_items.keys())[0],
        list(train_graph.item_to_users.keys())[0],
        train_graph
    )
    feature_size = len(sample_features)
    
    print(f"Feature size: {feature_size}")
    
    # Create model
    model = get_predictor_net(feature_size)
    model = model.to(device)
    print(f"Model created with {feature_size} input features")
    
    # Loss and optimizer
    # Model has 4 outputs for classes 0 (none), 1 (view), 2 (save), 3 (buy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load validation answers
    print("Loading validation answers...")
    val_labels = load_validation_answers(val_answers_path)
    
    # Create validation dataloader
    print("Creating validation dataloader...")
    val_dataset = ValDataset(val_queries_path, train_graph, feature_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_accuracy, predictions = evaluate(model, val_loader, val_labels, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")
    
    return model, predictions


if __name__ == '__main__':
    import sys
    
    # Set paths
    curr_dir = os.path.dirname(__file__)
    train_path = os.path.join(curr_dir, "../../../data/task1_train.tsv")
    val_queries_path = os.path.join(curr_dir, "../../../data/task1_val_queries.tsv")
    val_answers_path = os.path.join(curr_dir, "../../../data/task1_val_answers.tsv")
    
    # Check if MPS is available
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train and evaluate
    model, predictions = train_and_evaluate(
        train_path,
        val_queries_path,
        val_answers_path,
        num_epochs=20,
        batch_size=64,
        learning_rate=0.001,
        device=device
    )

