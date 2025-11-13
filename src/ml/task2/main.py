import argparse
import pandas as pd
import numpy as np

from graph import UserItemNetwork, pixie_recommend


def get_data(train_path, val_path, val_ans_path):
    train = pd.read_csv(
        train_path,
        names=["user", "item", "interaction"],
        sep="\t"
    )
    val = pd.read_csv(
        val_path,
        names=["user"],
        sep="\t"
    )
    val_answers = pd.read_csv(
        val_ans_path,
        names=["user", "item", "interaction"],
        sep="\t"
    )
    return train, val, val_answers


def build_gt_dict(val_answers: pd.DataFrame):
    """
    user -> set(items) 로 ground-truth 뷰 아이템을 만듦
    (val_answers는 interaction=1이니까 그냥 전부 넣으면 됨)
    """
    gt = {}
    for u, group in val_answers.groupby("user"):
        gt[u] = set(group["item"].tolist())
    return gt


def dcg_weights(k: int = 50):
    # w_j = 1 / log2(j+1)
    # j는 1부터 시작한다고 했으니 코드에서는 1..k
    weights = np.array([1.0 / np.log2(j + 1) for j in range(1, k + 1)], dtype=np.float64)
    return weights


def evaluate(predictions: dict, gt_dict: dict, k: int = 50) -> float:
    """
    predictions: user -> list of item_ids (length <= k)
    gt_dict: user -> set of ground-truth viewed items
    return: final score (mean of normalized scores)
    """
    weights = dcg_weights(k)
    max_score = weights.sum()  # 문제에서 모든 유저가 50개 있다고 했으니 공통
    scores = []

    for u, rec_items in predictions.items():
        gt_items = gt_dict.get(u, set())

        # S(u) = sum_{j} w_j * 1[rec_j in Gu]
        s_u = 0.0
        for j, item in enumerate(rec_items[:k]):  # j = 0..k-1
            if item in gt_items:
                s_u += weights[j]

        # normalized
        s_u_tilde = s_u / max_score if max_score > 0 else 0.0
        scores.append(s_u_tilde)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(
        description="Run Pixie recommendation on task2 data and evaluate on validation set."
    )
    parser.add_argument("--train-path", type=str,
                        default="../../../data/task2_train.tsv")
    parser.add_argument("--val-path", type=str,
                        default="../../../data/task2_val_queries.tsv")
    parser.add_argument("--val-answers-path", type=str,
                        default="../../../data/task2_val_answers.tsv")
    parser.add_argument("--steps", type=int, default=2000,
                        help="number of random-walk steps for Pixie")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="restart probability for Pixie")
    parser.add_argument("--topk", type=int, default=50,
                        help="how many items to recommend per user")
    args = parser.parse_args()

    # 1) load data
    train, val, val_answers = get_data(
        args.train_path, args.val_path, args.val_answers_path
    )

    # 2) build graph from train (uses interactions 2,3 inside)
    graph = UserItemNetwork(train)

    # 3) ground truth dict from val_answers
    gt_dict = build_gt_dict(val_answers)

    # 4) run pixie for every user in val
    predictions = {}
    val_users = val["user"].tolist()

    for u in val_users:
        recs = pixie_recommend(
            graph,
            user_raw_id=u,
            n_steps=args.steps,
            alpha=args.alpha,
            topk=args.topk,
        )
        predictions[u] = recs

    # 5) evaluate
    final_score = evaluate(predictions, gt_dict, k=args.topk)
    print(f"Final validation score: {final_score:.6f}")


if __name__ == "__main__":
    main()