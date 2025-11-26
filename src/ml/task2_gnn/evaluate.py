import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict

from ml.task2_gnn.model import HeteroSAGE
from ml.task2_gnn.data import load_dataset
from ml.task2_pixie.main import build_gt_dict, evaluate  # 위에서 만든 함수들

def build_idx2id(mapping):
    # {"raw_id": idx} -> ["idx": raw_id]
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_for_user(model, data, user_idx, num_items, topk, device, interacted_items=None):
    """
    단일 user (index 기준)에 대해 모든 item을 스코어링하고,
    상위 topk item index를 리턴.
    interacted_items: 이미 상호작용한 item index들의 집합/리스트 (추천 제외)
    """
    model.eval()
    with torch.no_grad():
        # user_idx 하나에 대해 모든 item(0..num_items-1) 평가
        item_indices = torch.arange(num_items, device=device, dtype=torch.long)
        user_indices = torch.full_like(item_indices, fill_value=user_idx, device=device)

        logits = model(data, user_indices, item_indices)  # [num_items, 1]
        logits = logits.view(-1)                          # [num_items]
        scores = torch.sigmoid(logits)                    # 확률로 해석 (optional)

        # 이미 상호작용한 아이템 제외
        if interacted_items is not None and len(interacted_items) > 0:
            # device에 맞는 tensor로 변환
            # interacted_items가 set이나 list라고 가정
            exclude_mask = torch.tensor(list(interacted_items), device=device, dtype=torch.long)
            scores[exclude_mask] = -float('inf')

        topk_scores, topk_idx = torch.topk(scores, k=min(topk, num_items))
        # topk_idx: item index들
        return topk_idx.cpu().numpy()   # numpy로 돌려주기 (편의를 위해)

def main():
    train_path = "./data/task2_train.tsv"
    val_path = "./data/task2_val_queries.tsv"
    val_ans_path = "./data/task2_val_answers.tsv"
    checkpoint_path = "./models/hetero_sage_model.pt"
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

    for u_raw in val_users:
        if u_raw not in user2idx:
            predictions[u_raw] = []
            continue

        u_idx = user2idx[u_raw]

        topk_item_indices = gnn_recommend_for_user(
            model, data,
            user_idx=u_idx,
            num_items=num_items,
            topk=topk,
            device=device,
            interacted_items=user_interacted.get(u_idx, set())
        )

        rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
        predictions[u_raw] = rec_items_raw

    final_score = evaluate(predictions, gt_dict, k=topk)
    print(f"Final validation score (HeteroSAGE): {final_score:.6f}")

if __name__ == "__main__":
    main()