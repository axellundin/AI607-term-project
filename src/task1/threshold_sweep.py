from task1.data import load_dataset, load_test_dataset, load_validation_dataset, get_negative_samples
from task1.model import HeteroSAGE
from util.metrics import compute_MF1
from itertools import product
from tqdm import tqdm
from settings import *
import numpy as np
import torch
import os

def predict_with_individual_thresholds(logits, thres1, thres2, thres3):
    probs = torch.sigmoid(logits)  
    
    b0 = probs[:, 0] >= thres1 
    b1 = probs[:, 1] >= thres2 
    b2 = probs[:, 2] >= thres3 
    
    pred = b0.int() + b1.int() + b2.int()
    
    return pred
    
def evaluate_thresholds(logits, labels, thres1, thres2, thres3):
    predictions = predict_with_individual_thresholds(logits, thres1, thres2, thres3)
    stats = compute_MF1(predictions, labels)
    
    return {
        'accuracy': stats['accuracy'],
        'macro_f1': stats['macro_f1'],
        'stats': stats
    }

def grid_search_individual_thresholds(use_training_set=False, threshold_range=None, step=0.05):
    # Load the saved model
    model_path = os.path.join(models_dir, "hetero_sage_model_approach9.pt")
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

    # Load data for message passing
    print("\nLoading training graph data...")
    train_data, _, _, train_labels_dict = load_dataset(training_data_filename)
    train_data = train_data.to(device)

    # Load dataset (training or validation)
    if use_training_set:
        print("Using TRAINING set.")
        # Generate negative samples to match training distribution 
        num_positive = len(train_labels_dict)
        print(f"Generating {num_positive} negative samples.")
        negative_labels = get_negative_samples(train_labels_dict, user2idx, item2idx, num_positive)
        
        # Combine samples
        pairs = list(train_labels_dict.keys()) + list(negative_labels.keys())
        labels = [train_labels_dict[pair] for pair in train_labels_dict.keys()] + [negative_labels[pair] for pair in negative_labels.keys()]
        dataset_name = "training"
        
        print(f"Total training set size: {len(pairs)}")
    else:
        print("Using VALIDATION set.")
        val_data_dict = load_validation_dataset(val_data_filename)
        pairs = list(val_data_dict.keys())
        labels = [val_data_dict[pair] for pair in pairs]
        dataset_name = "validation"

    labels_tensor = torch.tensor(labels)
    print(f"{dataset_name.capitalize()} set size: {len(pairs)}")

    # Generate predictions
    print(f"\nGenerating logits for {dataset_name} set.")
    batch_size = 16384
    logits_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Computing logits"):
            batch = pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in batch], device=device)
            item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in batch], device=device)
            
            # Get logits
            logits = model.predict(train_data, user_ids, item_ids) 
            logits_list.append(logits.cpu())

    # Concatenate all logits
    all_logits = torch.cat(logits_list) 
    print(f"Logits shape: {all_logits.shape}")

    # Set threshold range
    if threshold_range is None:
        threshold_range = (0.1, 0.85)
    threshold_min, threshold_max = threshold_range
    threshold_values = np.arange(threshold_min, threshold_max + step/2, step)

    # Generate combinations of thresholds
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

    # Generate new combinations of thresholds
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

def generate_final_predictions_on_test_set(thres1, thres2, thres3, model_filename="task1.pt", target_filename="task1_test_answers.tsv"):
    model_path = os.path.join(models_dir, model_filename)
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
    user2idx = checkpoint['user2idx']
    item2idx = checkpoint['item2idx']

    # Initialize model
    device = torch.device('cpu')
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Load data for message passing
    print("\nLoading training graph data...")    
    train_data, _, _, train_labels_dict = load_dataset(training_data_filename)
    train_data = train_data.to(device)
    
    pairs = load_test_dataset(test_data_filename)

    # Generate predictions
    print(f"\nGenerating predictions for test set.")
    print(f"Using thresholds {thres1}, {thres2}, {thres3}")
    batch_size = 16384
    final_predictions = torch.tensor([])

    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="Computing logits"):
            batch = pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in batch], device=device)
            item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in batch], device=device)
            
            # Get logits
            logits = model.predict(train_data, user_ids, item_ids) 
            predictions =  predict_with_individual_thresholds(logits, thres1, thres2, thres3)
            final_predictions = torch.concat([final_predictions, predictions.cpu()])

    target_file_path = os.path.join(results_dir, target_filename)
    print(f"Saving predictions to {target_file_path}")
    with open(target_file_path, "w") as f: 
        for (uid, iid), pred in zip(pairs, final_predictions):
            f.write(f"{uid}\t{iid}\t{int(pred)}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "individual":
            # Run individual threshold grid search
            use_training = len(sys.argv) > 2 and sys.argv[2] == "train"
            grid_search_individual_thresholds(use_training_set=use_training)
        elif sys.argv[1] == "train" and len(sys.argv) > 2 and sys.argv[2] == "individual":
            # Run individual threshold grid search on training set
            grid_search_individual_thresholds(use_training_set=True)
        elif sys.argv[1] == "test" and len(sys.argv) >= 5:
            thres_1 = float(sys.argv[2])
            thres_2 = float(sys.argv[3])
            thres_3 = float(sys.argv[4])
            generate_final_predictions_on_test_set(thres_1, thres_2, thres_3)
        else:
            print("Usage:")
            print("  python threshold_sweep.py                    # Run single threshold sweep")
            print("  python threshold_sweep.py individual        # Run individual threshold grid search on validation set")
            print("  python threshold_sweep.py individual train  # Run individual threshold grid search on training set")
            print("  python threshold_sweep.py individual train  # Run individual threshold grid search on training set")
            print("  python threshold_sweep.py test [view threshold] [save threshold] [buy threshold] # Compute final predictions")
    else:
        print("Usage:")
        print("  python threshold_sweep.py                    # Run single threshold sweep")
        print("  python threshold_sweep.py individual        # Run individual threshold grid search on validation set")
        print("  python threshold_sweep.py individual train  # Run individual threshold grid search on training set")
        print("  python threshold_sweep.py test [view threshold] [save threshold] [buy threshold] # Compute final predictions")

