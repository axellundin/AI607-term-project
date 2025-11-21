import os
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .evaluation import Evaluator
from datetime import datetime

class Trainer:
    """
    Generalized trainer for heterogeneous GNN models.
    """

    def __init__(self, model: nn.Module, data: HeteroData, device: torch.device,
                 optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 batch_size: int = 8192, checkpoint_interval: int = 10,
                 models_dir: str = './models', model_name: str = 'model'):
        self.model = model
        self.data = data
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.models_dir = models_dir
        self.model_name = model_name

        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)

    def train_epoch(self, train_pairs: List[Tuple[str, str]], train_labels: List[int],
                    user2idx: Dict[str, int], item2idx: Dict[str, int]) -> float:
        """
        Train for one epoch.

        Args:
            train_pairs: List of (user_id, item_id) tuples
            train_labels: List of labels
            user2idx: User ID to index mapping
            item2idx: Item ID to index mapping

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Shuffle training data
        perm = torch.randperm(len(train_pairs))

        total_loss = 0
        num_batches = 0

        # Batch training
        for i in range(0, len(train_pairs), self.batch_size):
            batch_indices = perm[i:i+self.batch_size]

            # Get batch data
            batch_pairs = [train_pairs[idx] for idx in batch_indices]
            batch_labels_raw = [train_labels[idx] for idx in batch_indices]

            # Convert IDs to indices
            batch_user_ids = torch.tensor([user2idx[user_id] for user_id, _ in batch_pairs], device=self.device)
            batch_item_ids = torch.tensor([item2idx[item_id] for _, item_id in batch_pairs], device=self.device)
            batch_labels = torch.tensor(batch_labels_raw, device=self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(self.data, batch_user_ids, batch_item_ids)
            loss = self.criterion(logits, batch_labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, user2idx: Dict[str, int], item2idx: Dict[str, int],
                       hyperparameters: Dict[str, Any]):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d")
        checkpoint_path = os.path.join(self.models_dir, f"{self.model_name}_{timestamp}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user2idx': user2idx,
            'item2idx': item2idx,
            'hyperparameters': hyperparameters
        }, checkpoint_path)
        print(f"  Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, Dict[str, int], Dict[str, int], Dict[str, Any]]:
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint.get('epoch', 0)
            user2idx = checkpoint.get('user2idx', {})
            item2idx = checkpoint.get('item2idx', {})
            hyperparameters = checkpoint.get('hyperparameters', {})

            print(f"Resuming training from epoch {start_epoch + 1}")
            print("Model and optimizer states loaded successfully")

            return start_epoch, user2idx, item2idx, hyperparameters
        else:
            print("No checkpoint found. Starting training from scratch.")
            return 0, {}, {}, {}

    def train(self, train_pairs: List[Tuple[str, str]], train_labels: List[int],
              user2idx: Dict[str, int], item2idx: Dict[str, int],
              val_pairs: List[Tuple[str, str]], val_labels: List[int],
              num_epochs: int, hyperparameters: Dict[str, Any],
              resume: bool = True) -> int:
        """
        Train the model for multiple epochs.

        Args:
            train_pairs: Training pairs
            train_labels: Training labels
            user2idx: User mapping
            item2idx: Item mapping
            val_pairs: Validation pairs
            val_labels: Validation labels
            num_epochs: Number of epochs
            hyperparameters: Model hyperparameters
            resume: Whether to resume from checkpoint

        Returns:
            Last epoch trained
        """
        # Create evaluator
        evaluator = Evaluator(self.model, self.data, self.device, user2idx, item2idx, self.batch_size)

        start_epoch = 0
        best_accuracy = 0.0
        best_path = None

        if resume:
            # Find the latest checkpoint file that starts with model_name
            matching_files = [f for f in os.listdir(self.models_dir) if f.startswith(f"{self.model_name}.pt")]
            if matching_files:
                # Sort by modification time, take the latest
                matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.models_dir, x)), reverse=True)
                latest_file = matching_files[0]
                checkpoint_path = os.path.join(self.models_dir, latest_file)
                start_epoch, _, _, _ = self.load_checkpoint(checkpoint_path)
                # Evaluate the loaded model on val to set best_accuracy
                metrics = evaluator.evaluate(val_pairs, val_labels)
                best_accuracy = metrics['accuracy']
                best_path = checkpoint_path
                print(f"Resumed from {checkpoint_path}, current best accuracy: {best_accuracy:.4f}")
            else:
                print(f"No checkpoint found for {self.model_name}. Starting from scratch.")

        print("Starting Training!")
        print("=" * 60)

        last_epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):
            avg_loss = self.train_epoch(train_pairs, train_labels, user2idx, item2idx)

            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

            # Evaluate on validation
            metrics = evaluator.evaluate(val_pairs, val_labels)
            accuracy = metrics['accuracy']
            print(f"Validation Accuracy: {accuracy:.4f}")

            # Save if best
            if accuracy > best_accuracy:
                # Delete old best
                if best_path and os.path.exists(best_path):
                    os.remove(best_path)
                # Save new best
                best_path = self.save_checkpoint(epoch + 1, user2idx, item2idx, hyperparameters)
                best_accuracy = accuracy
                print(f"New best model saved with accuracy {best_accuracy:.4f}")

            last_epoch = epoch + 1

        print("=" * 60)
        print("Training complete!")
        if best_path:
            print(f"Best model saved to {best_path} with accuracy {best_accuracy:.4f}")
        else:
            print("No model saved (no improvement on validation)")

        return last_epoch


def create_trainer(model: nn.Module, data: HeteroData, device: torch.device,
                   learning_rate: float = 0.01, batch_size: int = 8192,
                   models_dir: str = './models', model_name: str = 'model') -> Trainer:
    """
    Create a trainer with default optimizer and criterion.

    Args:
        model: The model to train
        data: The graph data
        device: Device to train on
        learning_rate: Learning rate
        batch_size: Batch size
        models_dir: Directory to save models
        model_name: Name for the model file

    Returns:
        Trainer instance
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    return Trainer(model, data, device, optimizer, criterion, batch_size,
                   models_dir=models_dir, model_name=model_name)