import torch
import numpy as np
import os

# Import the lightweight model class and FullGraph
from pyg_train_lightweight import LightweightUserItemGNN
from util.graphs import FullGraph

curr_dir = os.path.dirname(__file__)

def load_model_and_mappings(model_path="user_item_gnn_best.pt"):
    """Load the trained lightweight model and build the graph with mappings"""
    # Load the graph from training data
    train_file = os.path.join(curr_dir, "../../data/task1_train.tsv")
    G = FullGraph(train_file)
    
    # Build PyG graph with same mappings as training (no negative sampling for eval)
    data, user2idx, item2idx = G.build_pyg_graph(num_negative_samples_per_pos=0)
    
    # Load the model
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    model_full_path = os.path.join(curr_dir, model_path)
    checkpoint = torch.load(model_full_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if 'val_f1' in checkpoint:
        print(f"Loading model from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Training Val Acc: {checkpoint.get('val_acc', 0):.4f}")
        print(f"  Training Val F1: {checkpoint.get('val_f1', 0):.4f}")
    
    # Get model architecture from checkpoint or use defaults
    hidden_dim = checkpoint.get('hidden_dim', 64)
    num_layers = checkpoint.get('num_layers', 2)
    
    # Initialize model with same architecture
    model = LightweightUserItemGNN(len(user2idx), len(item2idx), 
                                   hidden_dim=hidden_dim, 
                                   num_layers=num_layers).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get the edge index from training data (only positive edges for message passing)
    edge_index = data['user', 'interacts', 'item'].edge_index.to(device)
    
    return model, user2idx, item2idx, edge_index, device

def evaluate_on_validation(model_path="user_item_gnn_best.pt", 
                          queries_suffix="val", 
                          answers_suffix="val",
                          batch_size=16384):
    """Evaluate the model on validation or test data with mini-batches"""
    queries_file = os.path.join(curr_dir, f"../../data/task1_{queries_suffix}_queries.tsv")
    answers_file = os.path.join(curr_dir, f"../../data/task1_{answers_suffix}_answers.tsv")
    
    # Load model and mappings
    print("Loading model and building graph...")
    model, user2idx, item2idx, edge_index, device = load_model_and_mappings(model_path)
    print(f"  Users in training: {len(user2idx):,}")
    print(f"  Items in training: {len(item2idx):,}")
    
    # Load validation data
    print(f"\nLoading {queries_suffix} data...")
    queries = []
    with open(queries_file, 'r') as f:
        for line in f.readlines():
            user_id, item_id = line.strip().split("\t")
            queries.append((user_id, item_id))
    
    answers = []
    with open(answers_file, 'r') as f:
        for line in f.readlines():
            user_id, item_id, interaction = line.strip().split("\t")
            answers.append(int(interaction))
    
    print(f"Loaded {len(queries):,} queries")
    
    # ============= MEMORY-EFFICIENT BATCH PREDICTION =============
    print("\nPreparing predictions...")
    
    # Filter queries to only those with known users/items
    valid_indices = []
    batch_user_indices = []
    batch_item_indices = []
    predictions = np.zeros(len(queries), dtype=int)  # Default to 0 for unknown users/items
    
    for i, (user_id, item_id) in enumerate(queries):
        if user_id in user2idx and item_id in item2idx:
            valid_indices.append(i)
            batch_user_indices.append(user2idx[user_id])
            batch_item_indices.append(item2idx[item_id])
    
    print(f"Valid queries (known users/items): {len(valid_indices):,} ({100*len(valid_indices)/len(queries):.1f}%)")
    print(f"Unknown user/item queries: {len(queries) - len(valid_indices):,}")
    
    # Process in batches to avoid OOM
    if len(valid_indices) > 0:
        print(f"Running batch predictions ({batch_size:,} per batch)...")
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(valid_indices), batch_size):
                end_idx = min(i + batch_size, len(valid_indices))
                
                batch_users = batch_user_indices[i:end_idx]
                batch_items = batch_item_indices[i:end_idx]
                
                edge_label_index = torch.tensor(
                    [batch_users, batch_items], 
                    dtype=torch.long
                ).to(device)
                
                logits = model(edge_index, edge_label_index)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_predictions.extend(preds)
        
        # Fill in predictions for valid queries
        for idx, pred in zip(valid_indices, all_predictions):
            predictions[idx] = pred
    # ======================================================
    
    # Initialize per-class metrics for F1-macro calculation
    TP = np.zeros(4)
    FP = np.zeros(4)
    FN = np.zeros(4)
    
    # Evaluate
    print("\nComputing metrics...")
    for predicted_label, true_label in zip(predictions, answers):
        for i in range(4):
            if true_label == i and predicted_label == i:
                TP[i] += 1
            elif true_label != i and predicted_label == i:
                FP[i] += 1
            elif true_label == i and predicted_label != i:
                FN[i] += 1
    
    # Compute per-class precision, recall, and F1 score
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    F1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Compute accuracy
    correct = (predictions == np.array(answers)).sum()
    accuracy = correct / len(answers)
    
    # Print detailed metrics
    print("\n" + "="*70)
    print(f"{queries_suffix.upper()} RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct:,}/{len(answers):,})")
    print("\nPer-class metrics:")
    class_names = ["No Interaction", "View", "Save", "Buy"]
    for i in range(4):
        support = TP[i] + FN[i]
        print(f"Class {i} ({class_names[i]:15s}): "
              f"Support={support:6.0f}, "
              f"TP={TP[i]:6.0f}, FP={FP[i]:6.0f}, FN={FN[i]:6.0f}, "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={F1_score[i]:.4f}")
    
    # Macro-F1 is the average of per-class F1 scores
    Macro_F1_score = np.mean(F1_score)
    
    print(f"\n{'='*70}")
    print(f"Macro F1 Score: {Macro_F1_score:.4f}")
    print(f"{'='*70}\n")
    
    return Macro_F1_score, accuracy

def generate_test_predictions(model_path="user_item_gnn_best.pt", 
                             output_file="task1_test_predictions.tsv",
                             batch_size=16384):
    """Generate predictions for test set and save to file"""
    queries_file = os.path.join(curr_dir, "../../data/task1_test_queries.tsv")
    output_path = os.path.join(curr_dir, output_file)
    
    # Load model and mappings
    print("Loading model and building graph...")
    model, user2idx, item2idx, edge_index, device = load_model_and_mappings(model_path)
    
    # Load test queries
    print("\nLoading test queries...")
    queries = []
    with open(queries_file, 'r') as f:
        for line in f.readlines():
            user_id, item_id = line.strip().split("\t")
            queries.append((user_id, item_id))
    
    print(f"Loaded {len(queries):,} test queries")
    
    # Prepare batch prediction
    print("Preparing predictions...")
    valid_indices = []
    batch_user_indices = []
    batch_item_indices = []
    predictions = np.zeros(len(queries), dtype=int)
    
    for i, (user_id, item_id) in enumerate(queries):
        if user_id in user2idx and item_id in item2idx:
            valid_indices.append(i)
            batch_user_indices.append(user2idx[user_id])
            batch_item_indices.append(item2idx[item_id])
    
    print(f"Valid queries: {len(valid_indices):,} ({100*len(valid_indices)/len(queries):.1f}%)")
    
    # Batch prediction
    if len(valid_indices) > 0:
        print(f"Running predictions ({batch_size:,} per batch)...")
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(valid_indices), batch_size):
                end_idx = min(i + batch_size, len(valid_indices))
                
                batch_users = batch_user_indices[i:end_idx]
                batch_items = batch_item_indices[i:end_idx]
                
                edge_label_index = torch.tensor(
                    [batch_users, batch_items], 
                    dtype=torch.long
                ).to(device)
                
                logits = model(edge_index, edge_label_index)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_predictions.extend(preds)
        
        for idx, pred in zip(valid_indices, all_predictions):
            predictions[idx] = pred
    
    # Save predictions
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, 'w') as f:
        for (user_id, item_id), pred in zip(queries, predictions):
            f.write(f"{user_id}\t{item_id}\t{pred}\n")
    
    print("Done!")
    return predictions

if __name__ == "__main__":
    import sys
    
    # Check if model file exists
    model_file = "user_item_gnn_best.pt"
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    
    model_path = os.path.join(curr_dir, model_file)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first using pyg_train_lightweight.py")
        sys.exit(1)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    macro_f1, accuracy = evaluate_on_validation(model_file)
    
    # Optionally generate test predictions
    generate_test = input("\nGenerate test predictions? (y/n): ").strip().lower()
    if generate_test == 'y':
        generate_test_predictions(model_file)

