import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ml.task2_gnn.model import HeteroSAGE
from ml.task2_gnn.data import load_dataset
from ml.task2_pixie.main import build_gt_dict, evaluate
from settings import data_dir, models_task2_dir  # 위에서 만든 함수들

def build_idx2id(mapping):
    # {"raw_id": idx} -> ["idx": raw_id]
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_batch(model, data, user_indices, num_items, topk, device, user_interacted_dict=None, batch_size=100):
    """
    Batch evaluation: 여러 user들에 대해 모든 item을 스코어링하고,
    각 user별로 상위 topk item index를 리턴.
    user_indices: user index들의 리스트/텐서
    user_interacted_dict: {user_idx: set of item_indices} - 이미 상호작용한 아이템 제외용
    batch_size: 한 번에 처리할 user 수 (메모리 제한을 위해)
    Returns: {user_idx: topk_item_indices} 딕셔너리
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Process users in batches to avoid memory issues
        num_batches = (len(user_indices) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(user_indices), batch_size), desc="Processing user batches", total=num_batches):
            batch_end = min(batch_start + batch_size, len(user_indices))
            batch_user_indices = user_indices[batch_start:batch_end]
            num_batch_users = len(batch_user_indices)
            
            user_indices_tensor = torch.tensor(batch_user_indices, device=device, dtype=torch.long)
            
            # 모든 (user, item) 쌍 생성: [num_batch_users * num_items]
            user_indices_expanded = user_indices_tensor.repeat_interleave(num_items)  # [num_batch_users * num_items]
            item_indices_expanded = torch.arange(num_items, device=device, dtype=torch.long).repeat(num_batch_users)  # [num_batch_users * num_items]
            
            # 배치로 모든 쌍 평가
            logits = model(data, user_indices_expanded, item_indices_expanded)  # [num_batch_users * num_items, 1]
            logits = logits.view(-1)  # [num_batch_users * num_items]
            scores = torch.sigmoid(logits)  # 확률로 해석
            scores = scores.view(num_batch_users, num_items)  # [num_batch_users, num_items]
            
            # 이미 상호작용한 아이템 제외
            if user_interacted_dict is not None:
                for i, user_idx in enumerate(batch_user_indices):
                    if user_idx in user_interacted_dict and len(user_interacted_dict[user_idx]) > 0:
                        exclude_items = torch.tensor(list(user_interacted_dict[user_idx]), device=device, dtype=torch.long)
                        scores[i, exclude_items] = -float('inf')
            
            # 각 user별로 topk 선택
            topk_scores, topk_idx = torch.topk(scores, k=min(topk, num_items), dim=1)  # [num_batch_users, topk]
            
            # 결과를 딕셔너리에 추가
            for i, user_idx in enumerate(batch_user_indices):
                results[user_idx] = topk_idx[i].cpu().numpy()
    
    return results

def main():
    train_path = os.path.join(data_dir, "task2_train.tsv")
    val_path = os.path.join(data_dir,"task2_val_queries.tsv")
    val_ans_path = os.path.join(data_dir,"task2_val_answers.tsv")
    checkpoint_path = os.path.join(models_task2_dir, "hetero_sage_model_baseline.pt")
    topk = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, user2idx, item2idx, labels = load_dataset(os.path.basename(train_path))
    num_users = len(user2idx)
    num_items = len(item2idx)

    # 학습 데이터에 있는 interaction 정보 구축 (이미 본 아이템 제외용)
    user_interacted = defaultdict(set)
    if ('user', 'interact', 'item') in data.edge_index_dict:
        edge_index = data['user', 'interact', 'item'].edge_index
        src_users = edge_index[0].numpy()
        dst_items = edge_index[1].numpy()
        for u, i in zip(src_users, dst_items):
            user_interacted[u].add(i)

    data = data.to(device)

    # idx -> raw_id 매핑
    idx2user = build_idx2id(user2idx)
    idx2item = build_idx2id(item2idx)

    val = pd.read_csv(
        val_path,
        names=["user"],
        sep="\t",
        dtype={"user" : str}
    )
    val_answers = pd.read_csv(
        val_ans_path,
        names=["user", "item", "interaction"],
        sep="\t",
        dtype={"user" : str, "item" : str}
    )    

    gt_dict = build_gt_dict(val_answers)

    # 3) 모델 로드
    embedding_dim = 128
    hidden_channels = 64
    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("WARNING: checkpoint not found, using randomly initialized model.")

    predictions = {}
    val_users = val["user"].tolist()
    
    # Filter out users not in training set and collect their indices
    valid_user_indices = []
    valid_user_raws = []
    for u_raw in val_users:
        if u_raw in user2idx:
            valid_user_indices.append(user2idx[u_raw])
            valid_user_raws.append(u_raw)
        else:
            predictions[u_raw] = []
    
    # Batch evaluation for all valid users
    if len(valid_user_indices) > 0:
        # Adjust batch_size based on available memory
        # Smaller batch_size = less memory usage, but more iterations
        # For ~10k items, batch_size=100 means ~1M pairs per batch
        batch_size = 100
        print(f"Evaluating {len(valid_user_indices)} users in batches of {batch_size}...")
        batch_results = gnn_recommend_batch(
            model, data,
            user_indices=valid_user_indices,
            num_items=num_items,
            topk=topk,
            device=device,
            user_interacted_dict=user_interacted,
            batch_size=batch_size
        )
        
        # Convert results to raw IDs
        for u_raw, u_idx in zip(valid_user_raws, valid_user_indices):
            topk_item_indices = batch_results[u_idx]
            rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
            predictions[u_raw] = rec_items_raw

    final_score = evaluate(predictions, gt_dict, k=topk)
    print(f"Final validation score (HeteroSAGE): {final_score:.6f}")

if __name__ == "__main__":
    main()