from settings import *
import os
from collections import Counter

# TODO: 
# Here we should implement a function for 
# computing the evaluation metrics for 
# task 1 and task 2

def compute_interaction_distribution(filename=val_data_filename):
    """
    Computes the distribution of interaction classes in the given file.
    Targeted for Task 1 validation file (task1_val_answers.tsv).
    """
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    counts = Counter()
    total_count = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        interaction = int(parts[2])
                        counts[interaction] += 1
                        total_count += 1
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
        
    distribution = {k: v / total_count for k, v in counts.items()} if total_count > 0 else {}
    
    return {
        'counts': dict(counts),
        'distribution': distribution,
        'total': total_count
    }

def compute_MF1(val_preds, val_labels):
    import torch
    
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

if __name__=='__main__':
    res = compute_interaction_distribution()
    print(res)