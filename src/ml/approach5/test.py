import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ml.approach5.data import load_dataset, load_validation_dataset
from ml.approach5.model import HeteroSAGE
from util.metrics import compute_MF1
from settings import *
import torch
from tqdm import tqdm
import torch.nn.functional as F


def predict_with_threshold(logits, threshold=0.5):
    probs = torch.sigmoid(logits)       # shape (batch, 3)

    # Boolean indicators for each ordinal step
    b0 = probs[:, 0] > threshold   # predicts y > 0
    b1 = probs[:, 1] > threshold   # predicts y > 1
    b2 = probs[:, 2] > threshold   # predicts y > 2

    # CORN prediction = number of True values
    pred = b0.int() + b1.int() + b2.int()

    return pred

# def predict_with_threshold(logits, threshold=0.5):
#     """
#     Convert 3 logits to class predictions using threshold.
    
#     Args:
#         logits: Tensor of shape (batch_size, 3) - 3 logits for view, save, buy
#         threshold: Float threshold value (default 0.5)
    
#     Returns:
#         predictions: Tensor of shape (batch_size,) with class predictions (0, 1, 2, or 3)
#     """
#     # Apply sigmoid to get probabilities
#     probs = torch.sigmoid(logits)  # Shape: (batch_size, 3)
    
#     # Predict based on threshold with cascading logic:
#     # If p1 < λ: predict no_interaction (0)
#     # Else if p2 < λ: predict view (1)
#     # Else if p3 < λ: predict save (2)
#     # Else: predict buy (3)
#     predictions = torch.zeros(probs.shape[0], dtype=torch.long, device=logits.device)
    
#     # p1 >= threshold means at least view
#     mask1 = probs[:, 0] >= threshold
#     predictions[mask1] = 1  # view
    
#     # p2 >= threshold means at least save
#     mask2 = probs[:, 1] >= threshold
#     predictions[mask2] = 2  # save (overwrites view if applicable)
    
#     # p3 >= threshold means buy
#     mask3 = probs[:, 2] >= threshold
#     predictions[mask3] = 3  # buy (overwrites save if applicable)
    
#     return predictions

def run_test(threshold=0.5):
    # Load the saved model
    print("Loading model for Approach 5...")
    model_path = os.path.join(models_dir, "hetero_sage_model_approach5.pt")
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
    train_data, _, _, _ = load_dataset(training_data_filename)
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
            logits = model.predict(train_data, val_user_ids, val_item_ids)  # Shape: (batch_size, 3)
            preds = predict_with_threshold(logits, threshold=threshold)
            val_preds_list.append(preds.cpu())

    # Concatenate all predictions
    val_preds = torch.cat(val_preds_list)
    val_labels_tensor = torch.tensor(val_labels)

    # Compute metrics
    stats = compute_MF1(val_preds, val_labels_tensor)

    print("\n" + "=" * 60)
    print("APPROACH 5 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(f"Validation Accuracy: {stats['accuracy']:.4f}")
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
    run_test()

