import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.task2_prediction.data import load_combined_dataset, load_validation_dataset
from ml.task2_prediction.model import HeteroSAGE
from ml.task2_prediction.test import predict_with_threshold
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import numpy as np

def sweep_thresholds():
    """
    Perform parameter sweep on threshold λ for prediction.
    Tests threshold values from 0.1 to 0.9 with step 0.05.
    """
    # Load the saved model
    print("Loading model for Task2 Prediction...")
    model_path = os.path.join(models_dir, "hetero_sage_model_task2_prediction.pt")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract hyperparameters and mappings
    num_users = checkpoint['hyperparameters']['num_users']
    num_items = checkpoint['hyperparameters']['num_items']
    embedding_dim = checkpoint['hyperparameters']['embedding_dim']
    hidden_channels = checkpoint['hyperparameters']['hidden_channels']
    user2idx = checkpoint['user2idx']
    item2idx = checkpoint['item2idx']

    print(f"Model hyperparameters:")
    print(f"  - num_users: {num_users}")
    print(f"  - num_items: {num_items}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - hidden_channels: {hidden_channels}")

    # Initialize model
    device = torch.device('cpu')
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Load the training graph data (needed for message passing)
    print("\nLoading training graph data...")
    train_data, _, _, _, _, _ = load_combined_dataset()
    train_data = train_data.to(device)

    # Load validation dataset
    print("Loading validation dataset...")
    val_data_dict = load_validation_dataset(val_data_filename)
    val_pairs = list(val_data_dict.keys())
    val_labels = [val_data_dict[pair] for pair in val_pairs]
    val_labels_tensor = torch.tensor(val_labels)

    print(f"Validation set size: {len(val_pairs)}")

    # Generate predictions once (before threshold sweep)
    print("\nGenerating logits for validation set...")
    batch_size = 8192 * 2
    val_logits_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(val_pairs), batch_size), desc="Computing logits"):
            val_batch = val_pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            val_user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in val_batch], device=device)
            val_item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in val_batch], device=device)
            
            # Get logits
            logits = model.predict(train_data, val_user_ids, val_item_ids)  # Shape: (batch_size, 1)
            val_logits_list.append(logits.cpu())

    # Concatenate all logits
    val_logits = torch.cat(val_logits_list)  # Shape: (len(val_pairs), 1)
    print(f"Logits shape: {val_logits.shape}")
    
    # Convert labels to binary: 1 for view, 0 for no view
    val_labels_binary = (val_labels_tensor == 1).long()

    # Parameter sweep: test threshold values from 0.1 to 0.9 with step 0.05
    threshold_range = np.arange(0.1, 0.95, 0.05)  # 0.1, 0.15, 0.2, ..., 0.9
    results = []

    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)
    print(f"Testing {len(threshold_range)} threshold values...")
    print()

    best_threshold = None
    best_mf1 = -1.0
    best_stats = None

    for threshold in tqdm(threshold_range, desc="Sweeping thresholds"):
        # Predict with current threshold
        val_preds = predict_with_threshold(val_logits, threshold=threshold)
        
        # Compute binary classification metrics
        accuracy = (val_preds == val_labels_binary).float().mean().item()
        
        TP = ((val_preds == 1) & (val_labels_binary == 1)).sum().item()
        FP = ((val_preds == 1) & (val_labels_binary == 0)).sum().item()
        FN = ((val_preds == 0) & (val_labels_binary == 1)).sum().item()
        
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }
        
        # Store results
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'stats': stats
        })
        
        # Track best threshold based on F1 score
        if f1 > best_mf1:
            best_mf1 = f1
            best_threshold = threshold
            best_stats = stats
        
        # Print current result
        print(f"λ={threshold:.2f}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP RESULTS (Binary Classification)")
    print("=" * 60)
    print(f"\nBest threshold: λ = {best_threshold:.2f}")
    print(f"Best F1 Score: {best_mf1:.4f}")
    print(f"Best Accuracy: {best_stats['accuracy']:.4f}")
    print(f"Best Precision: {best_stats['precision']:.4f}")
    print(f"Best Recall: {best_stats['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {best_stats['TP']}, FP: {best_stats['FP']}")
    print(f"  FN: {best_stats['FN']}")
    
    # Print top 5 thresholds
    print("\n" + "-" * 60)
    print("Top 5 thresholds by F1 Score:")
    print("-" * 60)
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. λ={result['threshold']:.2f}: F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}, Prec={result['precision']:.4f}, Rec={result['recall']:.4f}")
    
    print("=" * 60)
    
    # Save results to file
    results_dir_path = os.path.join(results_dir, "task2_prediction_threshold_sweep.txt")
    with open(results_dir_path, 'w') as f:
        f.write("THRESHOLD SWEEP RESULTS (Binary Classification)\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nBest threshold: λ = {best_threshold:.2f}\n")
        f.write(f"Best F1 Score: {best_mf1:.4f}\n")
        f.write(f"Best Accuracy: {best_stats['accuracy']:.4f}\n")
        f.write(f"Best Precision: {best_stats['precision']:.4f}\n")
        f.write(f"Best Recall: {best_stats['recall']:.4f}\n")
        f.write("\nAll results:\n")
        f.write("-" * 60 + "\n")
        for result in sorted_results:
            f.write(f"λ={result['threshold']:.2f}: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, Prec={result['precision']:.4f}, Rec={result['recall']:.4f}\n")
    
    print(f"\nResults saved to: {results_dir_path}")
    
    return best_threshold, best_stats, results

if __name__ == "__main__":
    sweep_thresholds()

