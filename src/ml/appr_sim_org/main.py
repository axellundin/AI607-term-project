#!/usr/bin/env python3
"""
Main script for training and evaluating heterogeneous GNN models for user-item interaction prediction.

Right now supports training, evaluation, and prediction modes.
and integrates with task 1 only.

Usage:
- To train a model:
    python3 -m ml.appr_sim_org.main --mode train --dataset task1 --model SAGE --embedding_dim

- To evaluate a model:
    python3 -m ml.appr_sim_org.main --mode eval --dataset task1 --model_name hetero_sage_model

- To make predictions:
    python3 -m ml.appr_sim_org.main --mode predict --dataset task1 --model_name hetero_sage_model

Args:
    --mode: Operation mode - 'train', 'eval', or 'predict'
    --dataset: Dataset name (affects file names)
    --data_dir: Directory containing data files
    --interaction_types: List of interaction types
    --model: Model type - 'GAT', 'SAGE', or 'GCN'
    --embedding_dim: Embedding dimension
    --hidden_channels: Hidden channels
    --num_layers: Number of layers
    --dropout: Dropout rate
    --aggr: Aggregation method for SAGE
    --num_epochs: Number of training epochs
    --batch_size: Batch size
    --learning_rate: Learning rate
    --negative_ratio: Ratio of negative samples to positive samples
    --resume: Resume training from check --point
    --models_dir: Directory to save/load models
    --model_name: Model name for saving/loading
    --eval_file: Evaluation file (if different from default)

"""

import argparse
import os
import torch
from settings import *
from ml.appr_sim_org.data import HeteroGraphDataLoader
from ml.appr_sim_org.models import create_model
from ml.appr_sim_org.train import create_trainer
from ml.appr_sim_org.evaluation import Evaluator

def get_args_from_parser():
    parser = argparse.ArgumentParser(description='Train or evaluate heterogeneous GNN models')

    # Mode
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict'],
                       default='train', help='Mode: train, eval, or predict')

    # Dataset
    parser.add_argument('--dataset', type=str, default='task1',
                       help='Dataset name (affects file names) for \'task1\' or \'task2\'')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                       help='Directory containing data files')
    parser.add_argument('--interaction_types', type=str, nargs='+',
                       default=['view', 'save', 'buy'],
                       help='List of interaction types')

    # Model
    parser.add_argument('--model', type=str, choices=['GAT', 'SAGE', 'GCN', 'SAGEV2'],
                       default='SAGE', help='Model type')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--aggr', type=str, default='sum',
                       choices=['mean', 'sum', 'max'],
                       help='Aggregation method for SAGE')

    # Training
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8192,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='L2 regularization weight decay')
    parser.add_argument('--negative_ratio', type=float, default=1/3,
                       help='Ratio of negative samples to positive samples')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')

    # Paths
    parser.add_argument('--models_dir', type=str, default=models_dir,
                       help='Directory to save/load models')
    parser.add_argument('--model_name', type=str, default='hetero_model',
                       help='Model name for saving/loading')

    # Evaluation
    parser.add_argument('--eval_file', type=str, default=None,
                       help='Evaluation file (if different from default)')

    return parser.parse_args()

def main(args):
    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize data loader
    data_loader = HeteroGraphDataLoader(args.data_dir, args.interaction_types)

    if args.mode == 'train':
        # Load training data
        print("Loading training data...")
        data, user2idx, item2idx, labels = data_loader.load_dataset(f"{args.dataset}_train.tsv")

        # Add negative samples
        num_negative_samples = int(len(labels) * args.negative_ratio)
        negative_labels = data_loader.get_negative_samples(labels, user2idx, item2idx, num_negative_samples)
        labels.update(negative_labels)

        train_pairs = list(labels.keys())
        train_labels = [labels[pair] for pair in train_pairs]

        # Load validation data
        print("Loading validation data...")
        val_file = f"{args.dataset}_val_answers.tsv"
        val_data = data_loader.load_validation_dataset(val_file)
        val_pairs = list(val_data.keys())
        val_labels = [val_data[pair] for pair in val_pairs]

        num_users = len(user2idx)
        num_items = len(item2idx)
        num_classes = len(args.interaction_types) + 1

        print(f"Training on {len(train_pairs)} samples, validating on {len(val_pairs)} samples")
        print(f"Users: {num_users}, Items: {num_items}")

        # Create model
        model = create_model(
            args.model, num_users, num_items, args.embedding_dim,
            args.hidden_channels, args.num_layers, args.dropout, num_classes,
            aggr=args.aggr
        ).to(device)

        data = data.to(device)

        # Create trainer
        trainer = create_trainer(
            model, data, device, args.learning_rate, args.weight_decay, args.batch_size,
            args.models_dir, args.model_name, class_weights=data_loader.get_class_weights().to(device)
        )

        # Training hyperparameters
        hyperparameters = {
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': args.embedding_dim,
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            'model_type': args.model,
            'aggr': args.aggr,
            'num_classes': num_classes,
        }

        # Train
        trainer.train(train_pairs, train_labels, user2idx, item2idx,
                     val_pairs, val_labels, args.num_epochs, hyperparameters, args.resume)

    elif args.mode in ['eval', 'predict']:
        # Load model
        model_path = os.path.join(args.models_dir, f"{args.model_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Extract hyperparameters
        hyperparams = checkpoint['hyperparameters']
        num_users = hyperparams['num_users']
        num_items = hyperparams['num_items']
        embedding_dim = hyperparams['embedding_dim']
        hidden_channels = hyperparams['hidden_channels']
        num_layers = hyperparams.get('num_layers', 2)
        dropout = hyperparams.get('dropout', 0.3)
        model_type = hyperparams.get('model_type', 'SAGE')
        aggr = hyperparams.get('aggr', 'mean')
        num_classes = hyperparams.get('num_classes', 4)

        # Create model
        model = create_model(
            model_type, num_users, num_items, embedding_dim,
            hidden_channels, num_layers, dropout, num_classes,
            aggr=aggr
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load training graph (needed for message passing)
        data, user2idx, item2idx, _ = data_loader.load_dataset(f"{args.dataset}_train.tsv")
        data = data.to(device)

        # Create evaluator
        evaluator = Evaluator(model, data, device, checkpoint['user2idx'], checkpoint['item2idx'], args.batch_size)

        if args.mode == 'eval':
            # Load evaluation data
            eval_file = args.eval_file or f"{args.dataset}_val_answers.tsv"
            print(f"Loading evaluation data from {eval_file}...")
            eval_data = data_loader.load_validation_dataset(eval_file)
            eval_pairs = list(eval_data.keys())
            eval_labels = [eval_data[pair] for pair in eval_pairs]

            print(f"Evaluating on {len(eval_pairs)} samples...")

            # Evaluate
            metrics = evaluator.evaluate(eval_pairs, eval_labels)
            evaluator.print_results(metrics)

        elif args.mode == 'predict':
            # Load test queries
            test_file = args.eval_file or f"{args.dataset}_test_queries.tsv"
            print(f"Loading test queries from {test_file}...")
            test_queries = data_loader.load_test_queries(test_file)

            print(f"Making predictions on {len(test_queries)} queries...")

            # Predict
            predictions = evaluator.predict(test_queries)

            # Create predictions folder
            predicts_dir = os.path.join(results_dir, "predicts")
            os.makedirs(predicts_dir, exist_ok=True)

            # Generate timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save predictions
            output_file = os.path.join(predicts_dir, f"{args.dataset}_{args.model_name}_predictions_{timestamp}.txt")
            with open(output_file, 'w') as f:
                for (user_id, item_id), pred in zip(test_queries, predictions):
                    f.write(f"{user_id}\t{item_id}\t{pred}\n")

            print(f"Predictions saved to {output_file}")


def manually_specify_args(**kwargs):
    args = argparse.Namespace()
    
    # Set default values
    args.mode = 'train'
    args.dataset = 'task1'
    args.data_dir = data_dir
    args.interaction_types = ['view', 'save', 'buy']
    args.model = 'SAGE'
    args.embedding_dim = 128
    args.hidden_channels = 64
    args.num_layers = 2
    args.dropout = 0.3
    args.aggr = 'sum'
    args.num_epochs = 10
    args.batch_size = 8192
    args.learning_rate = 0.01
    args.weight_decay = 0.0
    args.negative_ratio = 1/3
    args.resume = False
    args.models_dir = models_dir
    args.model_name = 'hetero_model'
    args.eval_file = None
    
    # Override defaults with provided kwargs
    for key, value in kwargs.items():
        if not hasattr(args, key):
            print(f"Warning: Unknown argument '{key}'")
        setattr(args, key, value)
        
    return args 

if __name__ == '__main__':
    args = get_args_from_parser()
    main(args)

