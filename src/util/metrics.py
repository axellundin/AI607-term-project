from settings import *
import torch

# TODO: 
# Here we should implement a function for 
# computing the evaluation metrics for 
# task 1 and task 2

def compute_MF1(val_preds, val_labels):
    # model.eval()
    # val_preds_list = []
    
    # with torch.no_grad():
    #     for i in range(0, len(val_features), batch_size): 
    #         batch_features = val_features[i:i+batch_size]
            
    #         logits = model(batch_features, edge_index)
    #         preds = logits.argmax(dim=1).cpu()
    #         val_preds_list.append(preds)
    
    # val_preds = torch.cat(val_preds_list)
    
    # Accuracy
    val_acc = (val_preds == val_labels).float().mean().item()
    
    # Per-class metrics for Macro F1
    TP = torch.zeros(4)
    FP = torch.zeros(4)
    FN = torch.zeros(4)
    
    for i in range(4):
        TP[i] = ((val_preds == i) & (val_labels == i)).sum()
        FP[i] = ((val_preds == i) & (val_labels != i)).sum()
        FN[i] = ((val_preds != i) & (val_labels == i)).sum()
    
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    macro_f1 = f1.mean().item()
    
    # Return comprehensive statistics
    stats = {
        'accuracy': val_acc,
        'macro_f1': macro_f1,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'TP': TP.int().tolist(),
            'FP': FP.int().tolist(),
            'FN': FN.int().tolist(),
        }
    }

    return stats