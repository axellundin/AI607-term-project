import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.approach9.data import load_dataset
from ml.approach9.model import HeteroSAGE
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import torch.nn.functional as F

def predict_with_threshold(logits, threshold=0.35):
    probs = torch.sigmoid(logits)       # shape (batch, 3)

    # Boolean indicators for each ordinal step
    b0 = probs[:, 0] > threshold   # predicts y > 0
    b1 = probs[:, 1] > threshold   # predicts y > 1
    b2 = probs[:, 2] > threshold   # predicts y > 2

    # CORN prediction = number of True values
    pred = b0.int() + b1.int() + b2.int()

    return pred

def run_evaluate_train(threshold=0.5):
    # Load the saved model
    print("Loading model for Approach 9...")
    model_path = os.path.join(models_dir, "hetero_sage_model_approach9.pt")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
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
    print(f"  - threshold: {threshold}")

    # Initialize model
    device = torch.device('cpu')
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Load the training graph data (needed for message passing)
    print("\nLoading training graph data...")
    train_data, _, _, train_labels = load_dataset(training_data_filename)
    train_data = train_data.to(device)

    # Get training pairs and labels
    train_pairs = list(train_labels.keys())
    train_labels_list = [train_labels[pair] for pair in train_pairs]

    print(f"Training set size: {len(train_pairs)}")

    # Evaluate on training set
    print("\nEvaluating on training set...")
    batch_size = 8192 * 2
    train_preds_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(train_pairs), batch_size)):
            train_batch = train_pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            train_user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in train_batch], device=device)
            train_item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in train_batch], device=device)
            
            # Predict
            logits = model.predict(train_data, train_user_ids, train_item_ids)  # Shape: (batch_size, 3)
            preds = predict_with_threshold(logits, threshold=threshold)
            train_preds_list.append(preds.cpu())

    # Concatenate all predictions
    train_preds = torch.cat(train_preds_list)
    train_labels_tensor = torch.tensor(train_labels_list)

    # Compute metrics
    stats = compute_MF1(train_preds, train_labels_tensor)

    print("\n" + "=" * 60)
    print("APPROACH 9 TRAINING SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Training Accuracy: {stats['accuracy']:.4f}")
    print(f"Macro F1 Score (MF1): {stats['macro_f1']:.4f}")
    print("\nPer-Class Metrics:")
    print("-" * 60)
    class_names = ['No Interaction (0)', 'View (1)', 'Save (2)', 'Buy (3)']
    for i, class_name in enumerate(class_names):
        if i < len(stats['per_class']['precision']):
            print(f"\n{class_name}:")
            print(f"  Precision: {stats['per_class']['precision'][i]:.4f}")
            print(f"  Recall:    {stats['per_class']['recall'][i]:.4f}")
            print(f"  F1 Score:  {stats['per_class']['f1'][i]:.4f}")
            print(f"  TP: {stats['per_class']['TP'][i]}, FP: {stats['per_class']['FP'][i]}, FN: {stats['per_class']['FN'][i]}")
    print("=" * 60)
    
    return stats

if __name__ == "__main__":
    run_evaluate_train()

