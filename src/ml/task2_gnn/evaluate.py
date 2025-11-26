import os
import torch
import pandas as pd
import numpy as np

from ml.task2_gnn.model import HeteroSAGE
from ml.task2_gnn.data import load_dataset
from ml.task2_pixie.main import build_gt_dict, evaluate  # 위에서 만든 함수들

def build_idx2id(mapping):
    # {"raw_id": idx} -> ["idx": raw_id]
    idx2id = {idx: raw_id for raw_id, idx in mapping.items()}
    return idx2id

def gnn_recommend_for_user(model, data, user_idx, num_items, topk, device):
    """
    단일 user (index 기준)에 대해 모든 item을 스코어링하고,
    상위 topk item index를 리턴
    """
    model.eval()
    with torch.no_grad():
        # user_idx 하나에 대해 모든 item(0..num_items-1) 평가
        item_indices = torch.arange(num_items, device=device, dtype=torch.long)
        user_indices = torch.full_like(item_indices, fill_value=user_idx, device=device)

        logits = model(data, user_indices, item_indices)  # [num_items, 1]
        logits = logits.view(-1)                          # [num_items]
        scores = torch.sigmoid(logits)                    # 확률로 해석 (optional)

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
            device=device
        )

        rec_items_raw = [idx2item[int(i)] for i in topk_item_indices]
        predictions[u_raw] = rec_items_raw

    final_score = evaluate(predictions, gt_dict, k=topk)
    print(f"Final validation score (HeteroSAGE): {final_score:.6f}")

if __name__ == "__main__":
    main()