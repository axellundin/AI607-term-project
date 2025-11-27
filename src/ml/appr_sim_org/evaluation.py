import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .data import HeteroGraphDataLoader

class Evaluator:
    """
    Evaluator for heterogeneous GNN models.
    """

    def __init__(self, model: nn.Module, data: HeteroData, device: torch.device,
                 user2idx: Dict[str, int], item2idx: Dict[str, int],
                 batch_size: int = 8192, threshold_moving: bool = False,
                 class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.data = data
        self.device = device
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.batch_size = batch_size
        self.threshold_moving = threshold_moving
        self.class_weights = class_weights

    def apply_threshold_moving(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply threshold moving to logits by adjusting with class weights.

        Args:
            logits: Raw logits from model (batch_size, num_classes)

        Returns:
            Adjusted logits
        """
        if self.threshold_moving and self.class_weights is not None:
            # Apply threshold moving by adjusting logits with log(class_weights)
            # This moves decision boundaries based on class priors
            log_weights = torch.log(self.class_weights + 1e-8)  # Add small epsilon to avoid log(0)
            adjusted_logits = logits + log_weights.to(logits.device)
            return adjusted_logits
        return logits

    def evaluate(self, eval_pairs: List[Tuple[str, str]], eval_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model on given pairs and labels.

        Args:
            eval_pairs: List of (user_id, item_id) tuples
            eval_labels: List of ground truth labels

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i in tqdm(range(0, len(eval_pairs), self.batch_size), desc="Evaluating"):
                batch_pairs = eval_pairs[i:i+self.batch_size]
                batch_labels = eval_labels[i:i+self.batch_size]

                # Convert to indices (use 0 for unknown users/items)
                batch_user_ids = torch.tensor([self.user2idx.get(uid, 0) for uid, _ in batch_pairs], device=self.device)
                batch_item_ids = torch.tensor([self.item2idx.get(iid, 0) for _, iid in batch_pairs], device=self.device)

                # Predict
                logits = self.model.predict(self.data, batch_user_ids, batch_item_ids) # type: ignore
                # Apply threshold moving if enabled
                adjusted_logits = self.apply_threshold_moving(logits)
                preds = adjusted_logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(batch_labels)

        # Compute metrics
        return self.compute_metrics(np.array(all_preds), np.array(all_labels))

    def predict(self, test_queries: List[Tuple[str, str]]) -> List[int]:
        """
        Make predictions on test queries.

        Args:
            test_queries: List of (user_id, item_id) tuples

        Returns:
            List of predicted classes
        """
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for i in tqdm(range(0, len(test_queries), self.batch_size), desc="Predicting"):
                batch_queries = test_queries[i:i+self.batch_size]

                # Convert to indices (use 0 for unknown users/items)
                batch_user_ids = torch.tensor([self.user2idx.get(uid, 0) for uid, _ in batch_queries], device=self.device)
                batch_item_ids = torch.tensor([self.item2idx.get(iid, 0) for _, iid in batch_queries], device=self.device)

                # Predict
                logits = self.model.predict(self.data, batch_user_ids, batch_item_ids) # type: ignore
                # Apply threshold moving if enabled
                adjusted_logits = self.apply_threshold_moving(logits)
                preds = adjusted_logits.argmax(dim=-1).cpu().numpy()

                all_preds.extend(preds)

        return all_preds

    @staticmethod
    def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Args:
            preds: Predicted labels
            labels: Ground truth labels

        Returns:
            Dictionary with accuracy, macro F1, and per-class metrics
        """
        # Accuracy
        accuracy = (preds == labels).mean()

        # Per-class metrics
        num_classes = len(np.unique(labels))
        TP = np.zeros(num_classes)
        FP = np.zeros(num_classes)
        FN = np.zeros(num_classes)

        for i in range(num_classes):
            TP[i] = ((preds == i) & (labels == i)).sum()
            FP[i] = ((preds == i) & (labels != i)).sum()
            FN[i] = ((preds != i) & (labels == i)).sum()

        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        macro_f1 = f1.mean()

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'TP': TP.tolist(),
                'FP': FP.tolist(),
                'FN': FN.tolist(),
            }
        }

    def print_results(self, metrics: Dict[str, float], class_names: Optional[List[str]] = None):
        """
        Print evaluation results.

        Args:
            metrics: Metrics dictionary
            class_names: Names for classes
        """
        if class_names is None:
            class_names = ['No Interaction (0)', 'View (1)', 'Save (2)', 'Buy (3)']

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
        print("\nPer-Class Metrics:")
        print("-" * 60)
        for i, class_name in enumerate(class_names):
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['per_class']['precision'][i]:.4f}") # type: ignore
            print(f"  Recall:    {metrics['per_class']['recall'][i]:.4f}") # type: ignore
            print(f"  F1 Score:  {metrics['per_class']['f1'][i]:.4f}") # type: ignore
            print(f"  TP: {metrics['per_class']['TP'][i]}, FP: {metrics['per_class']['FP'][i]}, FN: {metrics['per_class']['FN'][i]}") # type: ignore
        print("=" * 60)