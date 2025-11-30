import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ml.task2_gnn.model import HeteroSAGE
import numpy as np

def load_checkpoint_and_extract_embeddings(ckpt_path, device, data):
    ckpt = torch.load(ckpt_path, map_location=device)

    hparams = ckpt["hyperparameters"]
    num_users = hparams["num_users"]
    num_items = hparams["num_items"]
    embedding_dim = hparams["embedding_dim"]
    hidden_channels = hparams["hidden_channels"]

    model = HeteroSAGE(num_users, num_items, embedding_dim, hidden_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    user_emb, item_emb = model.get_model_embedding(data)

    return user_emb.numpy(), item_emb.numpy()

def visualize_tsne(embeddings, labels, title="t-SNE", save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(7,7))

    # scatter with label colors
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap="coolwarm",   # user=0 (blue), item=1 (red)
        s=5,
        alpha=0.7
    )

    # legend
    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["User", "Item"], title="Node Type")

    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved at {save_path}")

    plt.show()


if __name__ == "__main__":
    ckpt_path = "./models/hetero_sage_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from ml.task2_gnn.data import load_joint_dataset
    data, u2i, i2i, labels, group1_view = load_joint_dataset("task1_train.tsv", "task2_train.tsv")
    data = data.to(device)


    user_emb, item_emb = load_checkpoint_and_extract_embeddings(ckpt_path, device, data)

    print("user_emb shape:", user_emb.shape)  
    print("item_emb shape:", item_emb.shape)

    all_emb = np.vstack([user_emb, item_emb])

    labels = np.array(
    [0] * user_emb.shape[0] +
    [1] * item_emb.shape[0])

    visualize_tsne(all_emb,labels,  title="User Embedding t-SNE", save_path="./ml/task2_gnn/results/tsne.png")