import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.approach8.data import load_combined_dataset, load_validation_dataset
from ml.approach8.model import HeteroSAGE
from ml.approach8.test import predict_from_corn_logits
import torch.nn.functional as F
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import numpy as np

def predict_class_with_threshold(logits, threshold=0.5):
    """
    CORN prediction with variable threshold: Convert 3 logits to class predictions.
    Each logit predicts P(y > k) for k = 0,1,2.
    
    Args:
        logits: Tensor of shape (batch_size, 3) - 3 logits for P(y > k)
        threshold: Float threshold value (default 0.5 for standard CORN)
    
    Returns:
        predictions: Tensor of shape (batch_size,) with class predictions (0, 1, 2, or 3)
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)  # Shape: (batch_size, 3)
    
    # CORN prediction: sum of probabilities > threshold
    # This yields class 0-3
    predictions = torch.sum(probs > threshold, dim=1)
    
    return predictions

def sweep_thresholds():
    """
    Perform parameter sweep on threshold λ for CORN prediction.
    Tests threshold values from 0.1 to 0.9 with step 0.05.
    Note: Standard CORN uses threshold=0.5, but we allow sweeps for experimentation.
    """
    # Load the saved model
    print("Loading model for Approach 7 (CORN)...")
    model_path = os.path.join(models_dir, "hetero_sage_model_approach8.pt")
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
    train_data, _, _, _, _, _ = load_combined_dataset(    
        task1_filename="task1_train.tsv",
     task2_filename="task2_train.tsv")
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
            logits = model.predict(train_data, val_user_ids, val_item_ids)  # Shape: (batch_size, 3)
            val_logits_list.append(logits.cpu())

    # Concatenate all logits
    val_logits = torch.cat(val_logits_list)  # Shape: (len(val_pairs), 3)
    print(f"Logits shape: {val_logits.shape}")

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
        # Predict with current threshold using CORN
        val_preds = predict_class_with_threshold(val_logits, threshold=threshold)
        
        # Compute metrics
        stats = compute_MF1(val_preds, val_labels_tensor)
        
        # Store results
        results.append({
            'threshold': threshold,
            'accuracy': stats['accuracy'],
            'macro_f1': stats['macro_f1'],
            'stats': stats
        })
        
        # Track best threshold based on macro F1
        if stats['macro_f1'] > best_mf1:
            best_mf1 = stats['macro_f1']
            best_threshold = threshold
            best_stats = stats
        
        # Print current result
        print(f"λ={threshold:.2f}: Accuracy={stats['accuracy']:.4f}, MF1={stats['macro_f1']:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 60)
    print(f"\nBest threshold: λ = {best_threshold:.2f}")
    print(f"Best Macro F1: {best_mf1:.4f}")
    print(f"Best Accuracy: {best_stats['accuracy']:.4f}")
    
    print("\nBest threshold per-class metrics:")
    print("-" * 60)
    class_names = ['No Interaction (0)', 'View (1)', 'Save (2)', 'Buy (3)']
    for i, class_name in enumerate(class_names):
        if i < len(best_stats['per_class']['precision']):
            print(f"\n{class_name}:")
            print(f"  Precision: {best_stats['per_class']['precision'][i]:.4f}")
            print(f"  Recall:    {best_stats['per_class']['recall'][i]:.4f}")
            print(f"  F1 Score:  {best_stats['per_class']['f1'][i]:.4f}")
    
    # Print top 5 thresholds
    print("\n" + "-" * 60)
    print("Top 5 thresholds by Macro F1:")
    print("-" * 60)
    sorted_results = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. λ={result['threshold']:.2f}: MF1={result['macro_f1']:.4f}, Acc={result['accuracy']:.4f}")
    
    print("=" * 60)
    
    # Save results to file
    results_dir_path = os.path.join(results_dir, "approach8_threshold_sweep.txt")
    with open(results_dir_path, 'w') as f:
        f.write("THRESHOLD SWEEP RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nBest threshold: λ = {best_threshold:.2f}\n")
        f.write(f"Best Macro F1: {best_mf1:.4f}\n")
        f.write(f"Best Accuracy: {best_stats['accuracy']:.4f}\n")
        f.write("\nAll results:\n")
        f.write("-" * 60 + "\n")
        for result in sorted_results:
            f.write(f"λ={result['threshold']:.2f}: Accuracy={result['accuracy']:.4f}, MF1={result['macro_f1']:.4f}\n")
    
    print(f"\nResults saved to: {results_dir_path}")
    
    return best_threshold, best_stats, results

if __name__ == "__main__":
    sweep_thresholds()

