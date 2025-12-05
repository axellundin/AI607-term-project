from settings import *
import os
from collections import Counter
import numpy as np

def compute_interaction_distribution(filename=val_data_filename):
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

def dcg_weights(k: int = 50):
    weights = np.array([1.0 / np.log2(j + 1) for j in range(1, k + 1)], dtype=np.float64)
    return weights

def evaluate_DCG(predictions: dict, gt_dict: dict, k: int = 50) -> float:
    weights = dcg_weights(k)
    max_score = weights.sum() 
    scores = []

    for u, rec_items in predictions.items():
        gt_items = gt_dict.get(u, set())

        s_u = 0.0
        for j, item in enumerate(rec_items[:k]): 
            if item in gt_items:
                s_u += weights[j]

        # normalized
        s_u_tilde = s_u / max_score if max_score > 0 else 0.0
        scores.append(s_u_tilde)

    if not scores:
        return 0.0
    return float(np.mean(scores))


if __name__=='__main__':
    res = compute_interaction_distribution()
    print(res)