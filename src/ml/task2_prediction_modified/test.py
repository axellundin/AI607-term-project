import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.task2_prediction.data import load_combined_dataset, load_validation_dataset
from ml.task2_prediction.model import HeteroSAGE
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import torch.nn.functional as F

def predict_with_threshold(logits, threshold=0.5):
    """
    Convert binary logit to class predictions using threshold.
    
    Args:
        logits: Tensor of shape (batch_size, 1) or (batch_size,) - single logit for view
        threshold: Float threshold value (default 0.5)
    
    Returns:
        predictions: Tensor of shape (batch_size,) with binary predictions (0 = no view, 1 = view)
    """
    # Apply sigmoid to get probabilities
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    probs = torch.sigmoid(logits)  # Shape: (batch_size,)
    
    # Binary prediction: 1 if probability >= threshold, 0 otherwise
    predictions = (probs >= threshold).long()
    
    return predictions

def run_test(threshold=0.5):
    # Load the saved model
    print("Loading model for Task2 Prediction...")
    model_path = os.path.join(models_dir, "hetero_sage_model_task2_prediction.pt")
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
    train_data, _, _, _, _, _ = load_combined_dataset()
    train_data = train_data.to(device)

    # Load validation dataset
    print("Loading validation dataset...")
    val_data_dict = load_validation_dataset(val_data_filename)
    val_pairs = list(val_data_dict.keys())
    val_labels = [val_data_dict[pair] for pair in val_pairs]

    print(f"Validation set size: {len(val_pairs)}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    batch_size = 8192 * 2
    val_preds_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(val_pairs), batch_size)):
            val_batch = val_pairs[i:i+batch_size]
            
            # Convert to indices (use 0 for unknown users/items)
            val_user_ids = torch.tensor([user2idx.get(uid, 0) for uid, _ in val_batch], device=device)
            val_item_ids = torch.tensor([item2idx.get(iid, 0) for _, iid in val_batch], device=device)
            
            # Predict
            logits = model.predict(train_data, val_user_ids, val_item_ids)  # Shape: (batch_size, 1)
            preds = predict_with_threshold(logits, threshold=threshold)
            val_preds_list.append(preds.cpu())

    # Concatenate all predictions
    val_preds = torch.cat(val_preds_list)
    val_labels_tensor = torch.tensor(val_labels)
    
    # Convert labels to binary: 1 for view, 0 for no view
    val_labels_binary = (val_labels_tensor == 1).long()

    # Compute binary classification metrics
    accuracy = (val_preds == val_labels_binary).float().mean().item()
    
    # Binary classification metrics
    TP = ((val_preds == 1) & (val_labels_binary == 1)).sum().item()
    FP = ((val_preds == 1) & (val_labels_binary == 0)).sum().item()
    FN = ((val_preds == 0) & (val_labels_binary == 1)).sum().item()
    TN = ((val_preds == 0) & (val_labels_binary == 0)).sum().item()
    
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    print("\n" + "=" * 60)
    print("TASK2 PREDICTION EVALUATION RESULTS (Binary Classification)")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {TP}, FP: {FP}")
    print(f"  FN: {FN}, TN: {TN}")
    print("=" * 60)
    
    stats = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    }
    
    return stats

if __name__ == "__main__":
    run_test()

