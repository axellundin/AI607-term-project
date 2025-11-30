import os

from torch.cpu import is_available
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # tested this, but does not work. 

from ml.task2_gnn.model import HeteroSAGE
from ml.task2_gnn.data import load_joint_dataset, load_validation_dataset, get_negative_samples
from torch_geometric.loader import LinkNeighborLoader
import torch 
from tqdm import tqdm
import numpy as np


CONFIG = {
    "training_data_user1": "task1_train.tsv",
    "training_data_user2": "task2_train.tsv",
    "val_data_filename":  "task2_val_answers.tsv",
    "num_epochs": 10,
    "embedding_dim": 128,
    "hidden_channels": 64,
    "batch_size": 8192 * 2,
    "learning_rate": 0.01,
    "models_dir": "./models",
    "checkpoint_name": "hetero_sage_model.pt",
    "neg_ratio": 1/3,          # positive 개수 대비 negative 비율
    "checkpoint_interval": 10,
}

def prepare_data(config, device):
    # joint graph (user1 + user2) + labels + user1 view 로그
    data, user2idx, item2idx, labels, group1_view = load_joint_dataset(
        config["training_data_user1"],
        config["training_data_user2"],
    )

    # negative sampling
    num_negative = int(len(labels) * config["neg_ratio"])
    neg_labels = get_negative_samples(labels, user2idx, item2idx, num_negative)
    labels.update(neg_labels)

    train_pairs = list(labels.keys())
    train_labels = [labels[p] for p in train_pairs]

    # validation (원하면 나중에 사용)
    val_dict = load_validation_dataset(config["val_data_filename"])
    val_pairs = list(val_dict.keys())
    val_labels = [val_dict[p] for p in val_pairs]

    data = data.to(device)

    return (
        data,
        user2idx,
        item2idx,
        train_pairs,
        train_labels,
        val_pairs,
        val_labels,
        group1_view,
    )

def build_model(config, num_users, num_items, device):
    model = HeteroSAGE(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
        hidden_channels=config["hidden_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    return model, optimizer, criterion

def load_checkpoint(model, optimizer, config, device):
    models_dir = config["models_dir"]
    ckpt_path = os.path.join(models_dir, config["checkpoint_name"])
    start_epoch = 0

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    return start_epoch

def save_checkpoint(model, optimizer, epoch, config, user2idx, item2idx):
    os.makedirs(config["models_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["models_dir"], config["checkpoint_name"])
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
            "hyperparameters": {
                "num_users": len(user2idx),
                "num_items": len(item2idx),
                "embedding_dim": config["embedding_dim"],
                "hidden_channels": config["hidden_channels"],
            },
        },
        ckpt_path,
    )
    print(f"  Checkpoint saved at epoch {epoch}")

def train_one_epoch(
    epoch,
    model,
    data,
    train_pairs,
    train_labels,
    user2idx,
    item2idx,
    optimizer,
    criterion,
    batch_size,
    device,
):
    model.train()
    perm = torch.randperm(len(train_pairs))
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch}")
    for i in pbar:
        batch_idx = perm[i : i + batch_size]

        batch_pairs = [train_pairs[j] for j in batch_idx]
        batch_labels_raw = [train_labels[j] for j in batch_idx]

        batch_user_ids = torch.tensor(
            [user2idx[u] for (u, _) in batch_pairs], device=device
        )
        batch_item_ids = torch.tensor(
            [item2idx[iid] for (_, iid) in batch_pairs], device=device
        )
        batch_labels = torch.tensor(batch_labels_raw, device=device)
        batch_labels = (batch_labels > 0).float()  # 2/3 -> 1, 0 -> 0

        optimizer.zero_grad()
        logits = model(data, batch_user_ids, batch_item_ids).view(-1)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(num_batches, 1)


def train():
    config = CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    (
        data,
        user2idx,
        item2idx,
        train_pairs,
        train_labels,
        val_pairs,
        val_labels,
        group1_view,
    ) = prepare_data(config, device)

    num_users = len(user2idx)
    num_items = len(item2idx)

    model, optimizer, criterion = build_model(config, num_users, num_items, device)

    start_epoch = load_checkpoint(model, optimizer, config, device)

    print("Starting Training!")
    print("=" * 60)

    for epoch in range(start_epoch, config["num_epochs"]):
        avg_loss = train_one_epoch(
            epoch + 1,
            model,
            data,
            train_pairs,
            train_labels,
            user2idx,
            item2idx,
            optimizer,
            criterion,
            batch_size=config["batch_size"],
            device=device,
        )
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % config["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, epoch + 1, config, user2idx, item2idx)

    print("=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    train()