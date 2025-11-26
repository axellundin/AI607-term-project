from ml.appr_sim_org.train import create_trainer
from ml.appr_sim_org.models import create_model
from ml.appr_sim_org.data import HeteroGraphDataLoader
from ml.appr_sim_org.evaluation import Evaluator
from settings import *
import torch
import os
import argparse

# Copied manually_specify_args logic from main.py to here to make it self-contained
# or we could import it if we refactored main.py to be importable without side effects.
# For now, let's reconstruct the training flow here using the settings.

def train_with_settings(settings):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize data loader
    data_loader = HeteroGraphDataLoader(settings.get('data_dir', data_dir), settings.get('interaction_types', ['view', 'save', 'buy']))

    # Load training data
    print("Loading training data...")
    data, user2idx, item2idx, labels = data_loader.load_dataset(f"{settings.get('dataset', 'task1')}_train.tsv")

    # Add negative samples
    num_negative_samples = int(len(labels) * settings.get('negative_ratio', 1/3))
    negative_labels = data_loader.get_negative_samples(labels, user2idx, item2idx, num_negative_samples)
    labels.update(negative_labels)

    train_pairs = list(labels.keys())
    train_labels = [labels[pair] for pair in train_pairs]

    # Load validation data
    print("Loading validation data...")
    val_file = f"{settings.get('dataset', 'task1')}_val_answers.tsv"
    val_data = data_loader.load_validation_dataset(val_file)
    val_pairs = list(val_data.keys())
    val_labels = [val_data[pair] for pair in val_pairs]

    num_users = len(user2idx)
    num_items = len(item2idx)
    num_classes = len(settings.get('interaction_types', ['view', 'save', 'buy'])) + 1

    print(f"Training on {len(train_pairs)} samples, validating on {len(val_pairs)} samples")
    print(f"Users: {num_users}, Items: {num_items}")

    # Create model
    model = create_model(
        settings.get('model', 'SAGE'), num_users, num_items, settings.get('embedding_dim', 128),
        settings.get('hidden_channels', 64), settings.get('num_layers', 2), settings.get('dropout', 0.3), num_classes,
        aggr=settings.get('aggr', 'sum')
    ).to(device)

    data = data.to(device)

    # Create trainer
    trainer = create_trainer(
        model, data, device, settings.get('learning_rate', 0.01), settings.get('weight_decay', 0.0), settings.get('batch_size', 8192),
        settings.get('models_dir', models_dir), settings.get('model_name', 'hetero_model'), class_weights=data_loader.get_class_weights().to(device)
    )

    # Training hyperparameters
    hyperparameters = {
        'num_users': num_users,
        'num_items': num_items,
        'embedding_dim': settings.get('embedding_dim', 128),
        'hidden_channels': settings.get('hidden_channels', 64),
        'num_layers': settings.get('num_layers', 2),
        'dropout': settings.get('dropout', 0.3),
        'weight_decay': settings.get('weight_decay', 0.0),
        'model_type': settings.get('model', 'SAGE'),
        'aggr': settings.get('aggr', 'sum'),
        'num_classes': num_classes,
    }

    # Train
    trainer.train(train_pairs, train_labels, user2idx, item2idx,
                    val_pairs, val_labels, settings.get('num_epochs', 10), hyperparameters, settings.get('resume', False))


def setup_and_train_large_GSAGE():
    settings = {
        "mode":'train',
        "dataset":'task1',
        "data_dir":data_dir,
        "interaction_types":['view', 'save', 'buy'],
        "model":'SAGEV2',
        "embedding_dim":128,
        "hidden_channels":64,
        "num_layers":2,
        "dropout":0.3,
        "aggr":'sum',
        "num_epochs":100,
        "batch_size":8192,
        "learning_rate":0.001,  
        "weight_decay":1e-5,   
        "negative_ratio":1/3,
        "resume":False,
        "models_dir":models_dir,
        "model_name":'hetero_model_better_regularization',
        "eval_file":None
    }
    train_with_settings(settings)

def setup_and_train_large_GAT():
    settings = {
        "mode":'train',
        "dataset":'task1',
        "data_dir":data_dir,
        "interaction_types":['view', 'save', 'buy'],
        "model":'GAT',
        "embedding_dim":256,
        "hidden_channels":128,
        "num_layers":2,
        "dropout":0.3,
        "aggr":'sum',
        "num_epochs":20,
        "batch_size":8192,
        "learning_rate":0.01,
        "negative_ratio":1/3,
        "resume":False,
        "models_dir":models_dir,
        "model_name":'hetero_model_GAT',
        "eval_file":None
    }
    train_with_settings(settings)

if __name__ == "__main__":
    # You can switch between these or add argparse to select one
    setup_and_train_large_GSAGE()

