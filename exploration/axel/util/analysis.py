import numpy as np
from clustering import load_clustering_cache, spectral_clustering, save_clustering_cache, create_cluster_mappings
from graphs import FullGraph 
import os
import networkx as nx
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(__file__)

G = FullGraph(os.path.join(curr_dir, "../../../data/task1_train.tsv"))

def plot_full_degree_distributions():
    user_deg_hist = G.get_degree_distribution_vector(G.user_to_items)
    item_deg_hist = G.get_degree_distribution_vector(G.item_to_users)

    # Create histogram with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # User degree distribution
    ax1.bar(range(len(user_deg_hist)), user_deg_hist)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('User Degree Distribution')
    ax1.grid(True, alpha=0.3)

    # Item degree distribution
    ax2.bar(range(len(item_deg_hist)), item_deg_hist)
    ax2.set_xlabel('Degree')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Item Degree Distribution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_filtered_degree_distribution():
    filtered_graphs = [(G.user_to_items_view, G.item_to_users_view),
                        (G.user_to_items_save, G.item_to_users_save),
                        (G.user_to_items_buy, G.item_to_users_buy)]
    labels = ["view", "save", "buy"]

    # Create histogram with 3 rows and 2 columns subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 8))

    for i, (u2i, i2u) in enumerate(filtered_graphs):
        user_deg_hist = G.get_degree_distribution_vector(u2i)
        item_deg_hist = G.get_degree_distribution_vector(i2u)
        
        # User degree distribution
        axes[i, 0].bar(range(len(user_deg_hist)), np.log(user_deg_hist + 1))
        axes[i, 0].set_xlabel('Degree')
        axes[i, 0].set_ylabel('Log Frequency')
        axes[i, 0].set_title(f'User Degree Distribution ({labels[i]})')
        axes[i, 0].grid(True, alpha=0.3)

        # Item degree distribution
        axes[i, 1].bar(range(len(item_deg_hist)), np.log(item_deg_hist + 1))
        axes[i, 1].set_xlabel('Degree')
        axes[i, 1].set_ylabel('Log Frequency')
        axes[i, 1].set_title(f'Item Degree Distribution ({labels[i]})')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_item_level_statistics():
    stats=[]
    for item_id in G.item_to_users.keys():
        avg, std, N = G.compute_consumer_group_statistics_item_level_partition(item_id)
        stats.append((avg, std, N)) 

    stats_sorted = sorted(stats, key=lambda x: x[2], reverse=True)
    avg_sorted = [x[0] for x in stats_sorted]
    std_sorted = [np.linalg.norm(x[1]) for x in stats_sorted]
    N_sorted = [x[2] for x in stats_sorted]

    # Item degree distribution
    plt.plot(N_sorted, std_sorted)
    plt.xlabel('N')
    plt.ylabel('std')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def perform_clustering(n_clusters=8):
    runs = [
        (G.user_to_items, G.item_to_users, "all"),
        (G.user_to_items_view, G.item_to_users_view, "view"),
        (G.user_to_items_save, G.item_to_users_save, "save"),
        (G.user_to_items_buy, G.item_to_users_buy, "buy"),
    ]
    for users, items, label in runs:
        # Cluster users based on item interactions 
        user_cluster_labels, user_to_labels = spectral_clustering(users, n_clusters)
        user_to_label, label_to_user = create_cluster_mappings(user_cluster_labels, user_to_labels)
        save_clustering_cache(user_to_label, label_to_user, "users", label, n_clusters)
        # Cluster items based on user interactions
        item_cluster_labels, item_to_labels = spectral_clustering(items, n_clusters)
        user_to_label, label_to_user = create_cluster_mappings(item_cluster_labels, item_to_labels)
        save_clustering_cache(user_to_label, label_to_user, "items", label, n_clusters)

def visualize_user_clusters(n_clusters):
    dir = os.path.join(os.path.dirname(__file__), "../clustering_cache")
    interaction = "buy"
    user_to_label, label_to_user, _ = load_clustering_cache("users", interaction, n_clusters, dir)
    item_to_label, label_to_item, _ = load_clustering_cache("items", interaction, n_clusters, dir)

    # Configure subplots to show all 8 plots (2 rows, 4 columns)
    num_user_clusters = len(label_to_user)
    num_item_clusters = len(label_to_item)

    fig, axes = plt.subplots(4, 5, figsize=(16, 8))
    axes = axes.flatten()

    # Get item labels for plotting and create mapping from cluster label to index
    item_cluster_labels = sorted(label_to_item.keys())
    cluster_label_to_idx = {label: idx for idx, label in enumerate(item_cluster_labels)}
    
    for idx, (user_label, users) in enumerate(sorted(label_to_user.items())):
        count = np.zeros(num_item_clusters)
        
        for user_id in users:
            # Get items this user interacted with
            if user_id in G.user_to_items:
                for item_id in G.user_to_items[user_id]:
                    if item_id in item_to_label:
                        item_cluster = item_to_label[item_id]
                        if item_cluster in cluster_label_to_idx:
                            count[cluster_label_to_idx[item_cluster]] += 1

        # Create bar plot
        ax = axes[idx]
        ax.bar(range(num_item_clusters), count, alpha=0.7)
        ax.set_xlabel('Item Cluster')
        ax.set_ylabel('Interaction Count')
        ax.set_title(f'User Cluster {user_label}\n({len(users)} users)')
        ax.set_xticks(range(num_item_clusters))
        ax.set_xticklabels(item_cluster_labels)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

def visualize_item_clusters(n_clusters):
    dir = os.path.join(os.path.dirname(__file__), "../clustering_cache")
    interaction = "buy"
    user_to_label, label_to_user, _ = load_clustering_cache("users", interaction, n_clusters, dir)
    item_to_label, label_to_item, _ = load_clustering_cache("items", interaction, n_clusters, dir)

    # Configure subplots to show all 8 plots (2 rows, 4 columns)
    num_user_clusters = len(label_to_user)
    num_item_clusters = len(label_to_item)
    print(f"{num_user_clusters=}")
    print(f"{num_item_clusters=}")
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 8))
    axes = axes.flatten()

    # Get user labels for plotting and create mapping from cluster label to index
    user_cluster_labels = sorted(label_to_user.keys())
    cluster_label_to_idx = {label: idx for idx, label in enumerate(user_cluster_labels)}
    
    for idx, (item_label, items) in enumerate(sorted(label_to_item.items())):
        count = np.zeros(num_user_clusters)
        
        for item_id in items:
            # Get users who interacted with this item
            if item_id in G.item_to_users:
                for user_id in G.item_to_users[item_id]:
                    if user_id in user_to_label:
                        user_cluster = user_to_label[user_id]
                        if user_cluster in cluster_label_to_idx:
                            count[cluster_label_to_idx[user_cluster]] += 1

        # Create bar plot
        ax = axes[idx]
        ax.bar(range(num_user_clusters), count, alpha=0.7)
        ax.set_xlabel('User Cluster')
        ax.set_ylabel('Interaction Count')
        ax.set_title(f'Item Cluster {item_label}\n({len(items)} items)')
        ax.set_xticks(range(num_user_clusters))
        ax.set_xticklabels(user_cluster_labels)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
        
def overlap_most_common_products():
    items = [(item_id,len(buyers)) for item_id, buyers in G.item_to_users.items()]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    
    most_popular_item_id = items_sorted[0][0]
    buyers_of_most_popular_item = G.item_to_users[most_popular_item_id]
    
    second_most_popular_item_id = items_sorted[1][0]
    buyers_of_second_most_popular_item = G.item_to_users[second_most_popular_item_id]

    third_most_popular_item_id = items_sorted[2][0]
    buyers_of_third_most_popular_item = G.item_to_users[third_most_popular_item_id]

    intersection12 = set(buyers_of_most_popular_item).intersection(buyers_of_second_most_popular_item)
    intersection13 = set(buyers_of_most_popular_item).intersection(buyers_of_third_most_popular_item)
    intersection23 = set(buyers_of_second_most_popular_item).intersection(buyers_of_third_most_popular_item)

    intersection123 = intersection12.intersection(intersection23) 

    print(f"Most popular item id: {most_popular_item_id}")
    print(f"Number buyers of most popular item: {len(buyers_of_most_popular_item)}")

    print(f"Second most popular item id: {second_most_popular_item_id}")
    print(f"Number buyers of second most popular item: {len(buyers_of_second_most_popular_item)}")
    
    print(f"Third most popular item id: {third_most_popular_item_id}")
    print(f"Number buyers of third most popular item: {len(buyers_of_third_most_popular_item)}")

    print(f"Intersection 1&2: {len(intersection12)}")
    print(f"Intersection 1&3: {len(intersection13)}")
    print(f"Intersection 2&3: {len(intersection23)}")
    print(f"Intersection 1&2&3: {len(intersection123)}")

def find_connected_components():
    print("Starting to create adj matrix")
    user_adj = G.get_k_in_common_adj_matrix(G.user_to_items_buy, k=10)
    print("Done!\nConverting to nx graph..")
    user_graph = nx.from_scipy_sparse_array(user_adj)
    print("Done!\nComputing n connected comp..")
    n = nx.number_connected_components(user_graph)

    print(f"Num conn components:{n}")

def buyer_count():
    items = [(item_id,len(buyers)) for item_id, buyers in G.item_to_users.items()]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True) 

    x = [x[1] for x in items_sorted[10:20]]
    print(x)

if __name__=='__main__':
    buyer_count()
    #overlap_most_common_products()
    # visualize_item_clusters(20)
    # visualize_user_clusters(20)
    # plt.show()