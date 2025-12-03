import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.approach10.data import load_dataset, load_validation_dataset, get_negative_samples
from ml.approach10.model import HeteroLightGCN
from ml.approach10.test import predict_with_threshold
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import numpy as np
from itertools import product

def predict_with_individual_thresholds(logits, thres1, thres2, thres3):
    """
    Predict using individual thresholds for each output dimension.
    
    Args:
        logits: Tensor of shape (batch_size, 3) - 3 logits for view, save, buy
        thres1: Threshold for first dimension (predicts y > 0)
        thres2: Threshold for second dimension (predicts y > 1)
        thres3: Threshold for third dimension (predicts y > 2)
    
    Returns:
        predictions: Tensor of shape (batch_size,) with class predictions (0, 1, 2, or 3)
    """
    probs = torch.sigmoid(logits)  # shape (batch, 3)
    
    # Boolean indicators for each ordinal step with individual thresholds
    b0 = probs[:, 0] > thres1  # predicts y > 0
    b1 = probs[:, 1] > thres2  # predicts y > 1
    b2 = probs[:, 2] > thres3  # predicts y > 2
    
    # CORN prediction = number of True values
    pred = b0.int() + b1.int() + b2.int()
    
    return pred
    

def evaluate_thresholds(logits, labels, thres1, thres2, thres3):
    """
    Evaluate accuracy and MF1 given individual thresholds for each output dimension.
    
    Args:
        logits: Tensor of shape (batch_size, 3) - model logits
        labels: Tensor of shape (batch_size,) - ground truth labels
        thres1: Threshold for first dimension (predicts y > 0)
        thres2: Threshold for second dimension (predicts y > 1)
        thres3: Threshold for third dimension (predicts y > 2)
    
    Returns:
        dict: Dictionary containing accuracy, macro_f1, and full stats
    """
    predictions = predict_with_individual_thresholds(logits, thres1, thres2, thres3)
    stats = compute_MF1(predictions, labels)
    
    return {
        'accuracy': stats['accuracy'],
        'macro_f1': stats['macro_f1'],
        'stats': stats
    }

def grid_search_individual_thresholds(use_training_set=False, threshold_range=None, step=0.05):
    """
    Perform grid search with individual thresholds for each output dimension.
    
    Args:
        use_training_set: If True, perform sweep on training set; otherwise use validation set
        threshold_range: Tuple (min, max) for threshold range. If None, uses (0.1, 0.85)
        step: Step size for threshold grid search (default 0.05)
    
    Returns:
        tuple: (best_thresholds, best_stats, results)
            - best_thresholds: Tuple (thres1, thres2, thres3) with best thresholds
            - best_stats: Dictionary with best metrics
            - results: List of all results
    """
    # Load the saved model
    print("Loading model for Approach 10...")
    model_path = os.path.join(models_dir, "hetero_sage_model_approach10.pt")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return None, None, None

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract hyperparameters and mappings
    num_users = checkpoint['hyperparameters']['num_users']
    num_items = checkpoint['hyperparameters']['num_items']
    embedding_dim = checkpoint['hyperparameters']['embedding_dim']
    hidden_channels = checkpoint['hyperparameters']['hidden_channels']
    num_layers = checkpoint['hyperparameters'].get('num_layers', 2)
    user2idx = checkpoint['user2idx']
    item2idx = checkpoint['item2idx']

    print(f"Model hyperparameters:")
    print(f"  - num_users: {num_users}")
    print(f"  - num_items: {num_items}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - hidden_channels: {hidden_channels}")
    print(f"  - num_layers: {num_layers}")

    # Initialize model
    device = torch.device('cpu')
    model = HeteroLightGCN(num_users, num_items, embedding_dim, hidden_channels, num_layers=num_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Load the training graph data (needed for message passing)
    print("\nLoading training graph data...")
    train_data, _, _, train_labels_dict = load_dataset(training_data_filename)
    train_data = train_data.to(device)

    # Load dataset (training or validation)
    if use_training_set:
        print("Using TRAINING set for threshold sweep...")
        # Generate negative samples to match training distribution (same amount as positive samples)
        num_positive = len(train_labels_dict)
        print(f"Training set has {num_positive} positive samples")
        print(f"Generating {num_positive} negative samples to match training distribution...")
        
        # Generate negative samples (excluding training positive interactions)
        negative_labels = get_negative_samples(train_labels_dict, user2idx, item2idx, num_positive)
        
        # Combine positive and negative samples
        pairs = list(train_labels_dict.keys()) + list(negative_labels.keys())
        labels = [train_labels_dict[pair] for pair in train_labels_dict.keys()] + [negative_labels[pair] for pair in negative_labels.keys()]
        dataset_name = "training"
        
        print(f"Total training set size (positives + negatives): {len(pairs)}")
    else:
        print("Loading VALIDATION dataset...")
        val_data_dict = load_validation_dataset(val_data_filename)
        pairs = list(val_data_dict.keys())
        labels = [val_data_dict[pair] for pair in pairs]
        dataset_name = "validation"

    labels_tensor = torch.tensor(labels)
    print(f"{dataset_name.capitalize()} set size: {len(pairs)}")

    # Generate predictions once (before threshold sweep)
    print(f"\nGenerating logits for {dataset_name} set...")
    batch_size = 8192 * 2
    logits_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Computing logits"):
            batch = pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in batch], device=device)
            item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in batch], device=device)
            
            # Get logits
            logits = model.predict(train_data, user_ids, item_ids)  # Shape: (batch_size, 3)
            logits_list.append(logits.cpu())

    # Concatenate all logits
    all_logits = torch.cat(logits_list)  # Shape: (len(pairs), 3)
    print(f"Logits shape: {all_logits.shape}")

    # Set threshold range
    if threshold_range is None:
        threshold_range = (0.1, 0.85)
    threshold_min, threshold_max = threshold_range
    threshold_values = np.arange(threshold_min, threshold_max + step/2, step)

    # Generate all combinations of thresholds
    threshold_combinations = list(product(threshold_values, threshold_values, threshold_values))
    total_combinations = len(threshold_combinations)
    
    print("\n" + "=" * 60)
    print("COARSE THRESHOLD GRID SEARCH")
    print("=" * 60)
    print(f"Dataset: {dataset_name.capitalize()} set")
    print(f"Threshold range: [{threshold_min:.2f}, {threshold_max:.2f}]")
    print(f"Step size: {step:.2f}")
    print(f"Total combinations: {total_combinations}")
    print()

    results = []
    best_thresholds = None
    best_mf1 = -1.0
    best_stats = None

    for thres1, thres2, thres3 in tqdm(threshold_combinations, desc="Grid searching thresholds"):
        # Evaluate with current thresholds
        result = evaluate_thresholds(all_logits, labels_tensor, thres1, thres2, thres3)
        
        # Store results
        results.append({
            'thres1': thres1,
            'thres2': thres2,
            'thres3': thres3,
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_f1'],
            'stats': result['stats']
        })
        
        # Track best thresholds based on macro F1
        if result['macro_f1'] > best_mf1:
            best_mf1 = result['macro_f1']
            best_thresholds = (thres1, thres2, thres3)
            best_stats = result['stats']

    step2 = 0.01
    range_0 = np.arange(best_thresholds[0] - step, best_thresholds[0] + step, step2)
    range_1 = np.arange(best_thresholds[1] - step, best_thresholds[1] + step, step2)
    range_2 = np.arange(best_thresholds[2] - step, best_thresholds[2] + step, step2)

    # Generate all combinations of thresholds
    threshold_combinations = list(product(range_0, range_1, range_2))
    total_combinations = len(threshold_combinations)
    
    print(f"\nBest thresholds: thres1={best_thresholds[0]:.2f}, thres2={best_thresholds[1]:.2f}, thres3={best_thresholds[2]:.2f}\n")
    print("\n" + "=" * 60)
    print("FINE THRESHOLD GRID SEARCH")
    print("=" * 60)
    print(f"Dataset: {dataset_name.capitalize()} set")
    print(f"Threshold range: [{threshold_min:.2f}, {threshold_max:.2f}]")
    print(f"Step size: {step:.2f}")
    print(f"Total combinations: {total_combinations}")
    print()

    results = []
    best_thresholds = None
    best_mf1 = -1.0
    best_stats = None

    for thres1, thres2, thres3 in tqdm(threshold_combinations, desc="Grid searching thresholds"):
        # Evaluate with current thresholds
        result = evaluate_thresholds(all_logits, labels_tensor, thres1, thres2, thres3)
        
        # Store results
        results.append({
            'thres1': thres1,
            'thres2': thres2,
            'thres3': thres3,
            'accuracy': result['accuracy'],
            'macro_f1': result['macro_f1'],
            'stats': result['stats']
        })
        
        # Track best thresholds based on macro F1
        if result['macro_f1'] > best_mf1:
            best_mf1 = result['macro_f1']
            best_thresholds = (thres1, thres2, thres3)
            best_stats = result['stats']

    # Print summary
    print("\n" + "=" * 60)
    print("INDIVIDUAL THRESHOLD GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"\nBest thresholds: thres1={best_thresholds[0]:.2f}, thres2={best_thresholds[1]:.2f}, thres3={best_thresholds[2]:.2f}")
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
    
    # Print top 5 threshold combinations
    print("\n" + "-" * 60)
    print("Top 5 threshold combinations by Macro F1:")
    print("-" * 60)
    sorted_results = sorted(results, key=lambda x: x['macro_f1'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. thres1={result['thres1']:.2f}, thres2={result['thres2']:.2f}, thres3={result['thres3']:.2f}: "
            f"MF1={result['macro_f1']:.4f}, Acc={result['accuracy']:.4f}")
    
    print("=" * 60)

    return best_thresholds, best_stats, results

def sweep_thresholds(lower_bound=0.1, upper_bound=0.95, step=0.05):
    """
    Perform parameter sweep on threshold λ for prediction.
    
    Args:
        lower_bound: Lower bound for threshold range (default 0.1)
        upper_bound: Upper bound for threshold range (default 0.95)
        step: Step size for threshold sweep (default 0.05)
    """
    # Load the saved model
    print("Loading model for Approach 10...")
    model_path = os.path.join(models_dir, "hetero_sage_model_approach10.pt")
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
    num_layers = checkpoint['hyperparameters'].get('num_layers', 2)
    user2idx = checkpoint['user2idx']
    item2idx = checkpoint['item2idx']

    print(f"Model hyperparameters:")
    print(f"  - num_users: {num_users}")
    print(f"  - num_items: {num_items}")
    print(f"  - embedding_dim: {embedding_dim}")
    print(f"  - hidden_channels: {hidden_channels}")
    print(f"  - num_layers: {num_layers}")

    # Initialize model
    device = torch.device('cpu')
    model = HeteroLightGCN(num_users, num_items, embedding_dim, hidden_channels, num_layers=num_layers).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Load the training graph data (needed for message passing)
    print("\nLoading training graph data...")
    train_data, _, _, train_labels_dict = load_dataset(training_data_filename)
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

    # Parameter sweep: test threshold values with specified range and step
    threshold_range = np.arange(lower_bound, upper_bound + step/2, step)
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
    results_dir_path = os.path.join(results_dir, "approach10_threshold_sweep.txt")
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
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "individual":
            # Run individual threshold grid search
            use_training = len(sys.argv) > 2 and sys.argv[2] == "train"
            grid_search_individual_thresholds(use_training_set=use_training)
        elif sys.argv[1] == "train" and len(sys.argv) > 2 and sys.argv[2] == "individual":
            # Run individual threshold grid search on training set
            grid_search_individual_thresholds(use_training_set=True)
        else:
            print("Usage:")
            print("  python threshold_sweep.py                    # Run single threshold sweep")
            print("  python threshold_sweep.py individual        # Run individual threshold grid search on validation set")
            print("  python threshold_sweep.py individual train  # Run individual threshold grid search on training set")
            sweep_thresholds()
    else:
        # Default: run single threshold sweep
        sweep_thresholds()
