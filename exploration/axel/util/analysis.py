import numpy as np
from clustering import load_clustering_cache, spectral_clustering, save_clustering_cache, create_cluster_mappings
from graphs import FullGraph 
import os
from sklearn.cluster import SpectralClustering
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

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

def fast_spectral_clustering():
    similarity_matrix = G.get_cosine_similarity_matrix(G.user_to_items_buy, filename="cosine_similarity_filtered")
    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42).fit_predict(similarity_matrix)
    user_list = list(G.user_to_items_buy.keys())

    cluster_sizes = np.bincount(labels)
    print("Cluster sizes:", cluster_sizes)

def fast_spectral_clustering():
    similarity_matrix = G.get_cosine_similarity_matrix(G.user_to_items_buy, filename="cosine_similarity_filtered")
    labels = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42).fit_predict(similarity_matrix)
    cluster_sizes = np.bincount(labels)
    print("Cluster sizes:", cluster_sizes)

def recursive_spectral_clustering(similarity_matrix, 
                                num_clusters=3, 
                                global_index_mapping=None, 
                                clusters=[],
                                max_depth=100,
                                depth=0, 
                                max_cluster_size= 1000,
                                min_cluster_size=100
                                ):
    if depth > max_depth or similarity_matrix.shape[0] < min_cluster_size:
        # Return all indices as a single cluster if max depth reached
        return [global_index_mapping]

    
    if global_index_mapping is None: 
        global_index_mapping = np.arange(similarity_matrix.shape[0])
        # Convert to CSR once at the beginning for sparse matrices
        if hasattr(similarity_matrix, 'tocsr'):
            similarity_matrix = similarity_matrix.tocsr()
    
    print("clustering similarity matrix of size n=", similarity_matrix.shape[0])
    
    # Skip clustering if matrix is too small
    if similarity_matrix.shape[0] < num_clusters:
        return [global_index_mapping]

    labels = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42).fit_predict(similarity_matrix)
    
    idx_results = []
    for i in range(num_clusters):
        idx = np.where(labels == i)[0]
        idx_results.append((idx, len(idx)))
    idx_sorted = sorted(idx_results, key=lambda x: x[1], reverse=True)

    print("divided into:", [x[1] for x in idx_sorted])
    i = num_clusters - 1
    while (idx_sorted[0][1] - idx_sorted[i][1]) / idx_sorted[0][1] > 0.2 and idx_sorted[i][1] < max_cluster_size:
        clusters.append(global_index_mapping[idx_sorted[i][0]])
        i -= 1
        if i == 0: break

    if i == num_clusters - 1 and idx_sorted[0][1] < max_cluster_size:
        print("in return clause")
        return [global_index_mapping[x[0]] for x in idx_sorted]
    
    for j in range(i+1): 
        idx = idx_sorted[j][0]
        new_global_index_mapping = global_index_mapping[idx]

        if hasattr(similarity_matrix, 'indices'):
            sub_similarity_matrix = similarity_matrix[idx, :][:, idx]
        else:
            sub_similarity_matrix = similarity_matrix[np.ix_(idx, idx)]
               
        new_clusters = recursive_spectral_clustering(sub_similarity_matrix, 2, new_global_index_mapping, [], max_depth, depth+1, max_cluster_size, min_cluster_size)

        del sub_similarity_matrix
        del new_global_index_mapping

        for cluster in new_clusters: 
            clusters.append(cluster)
        
    return clusters

def test_RSC_on_users():
    similarity_matrix = G.get_cosine_similarity_matrix(G.user_to_items, filename="cosine_similarity_all_interactions")
    clusters = recursive_spectral_clustering(similarity_matrix, num_clusters=2)
    print([len(x) for x in clusters])
    
    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "rsc_all_interactions_users.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {cache_file}")

def test_RSC_on_items():
    similarity_matrix = G.get_cosine_similarity_matrix(G.item_to_users, filename="cosine_similarity_all_interactions_items")
    clusters = recursive_spectral_clustering(similarity_matrix, num_clusters=2)
    print([len(x) for x in clusters])
    
    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "rsc_all_interactions_items.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {cache_file}")

def test_RSC_for_k_in_common():
    similarity_matrix = G.get_k_in_common_adj_matrix(G.user_to_items_buy, k=5)
    # Keep as sparse matrix - don't convert to dense
    clusters = recursive_spectral_clustering(similarity_matrix, num_clusters=10)
    print([len(x) for x in clusters])
    
    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "recursive_spectral_clustering_clusters.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {cache_file}")

def test_RSC_for_intersection_count_sim_on_users():
    similarity_matrix = G.get_intersection_similarity_matrix(G.user_to_items, filename="intersection_count_similarity_users")
    # Keep as sparse matrix - don't convert to dense
    clusters = recursive_spectral_clustering(similarity_matrix, num_clusters=2)
    print([len(x) for x in clusters])
    
    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_users.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {cache_file}")

def test_RSC_for_intersection_count_sim_on_items():
    similarity_matrix = G.get_intersection_similarity_matrix(G.item_to_users, filename="intersection_count_similarity_items")
    # Keep as sparse matrix - don't convert to dense
    clusters = recursive_spectral_clustering(similarity_matrix, num_clusters=2)
    print([len(x) for x in clusters])
    
    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_items.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(clusters, f)
    print(f"Clusters saved to {cache_file}")

def analyze_user_clusters():
    cache_dir = os.path.join(curr_dir, "../cache")
    try:
        cache_file_users = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_users.pkl")
    except:
        print("The desired files don't seem to exist.")
        return

    with open(cache_file_users, 'rb') as f:
        user_clusters = pickle.load(f)
    
    user_clusters_sorted = sorted(user_clusters, key=lambda x: len(x))

    idx_to_user_id = list(G.user_to_items.keys())

    for cluster_number, idx in enumerate(user_clusters_sorted): 
        user_ids = [idx_to_user_id[i] for i in idx]
        view_sum = 0
        save_sum = 0
        buy_sum = 0
        
        for user_id in user_ids:
            view_sum += len(G.user_to_items_view[user_id])
            save_sum += len(G.user_to_items_save[user_id])
            buy_sum += len(G.user_to_items_buy[user_id])

        view_sum /= len(user_ids)
        save_sum /= len(user_ids)
        buy_sum /= len(user_ids)

        print(f"Cluster nr:{cluster_number}\tSize:{len(user_ids)}")
        print(f"\tAvg views:\t{int(view_sum)}")
        print(f"\tAvg saves:\t{int(save_sum)}")
        print(f"\tAvg buys:\t{int(buy_sum)}")

def analyze_item_clusters():
    cache_dir = os.path.join(curr_dir, "../cache")
    try:
        cache_file_items = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_items.pkl")
    except:
        print("The desired files don't seem to exist.")
        return

    with open(cache_file_items, 'rb') as f:
        item_clusters = pickle.load(f)
    
    item_clusters_sorted = sorted(item_clusters, key=lambda x: len(x))

    idx_to_item_id = list(G.item_to_users.keys())

    for cluster_number, idx in enumerate(item_clusters_sorted): 
        item_ids = [idx_to_item_id[i] for i in idx]
        view_sum = 0
        save_sum = 0
        buy_sum = 0
        
        for item_id in item_ids:
            view_sum += len(G.item_to_users_view[item_id])
            save_sum += len(G.item_to_users_save[item_id])
            buy_sum += len(G.item_to_users_buy[item_id])

        view_sum /= len(item_ids)
        save_sum /= len(item_ids)
        buy_sum /= len(item_ids)

        print(f"Cluster nr:{cluster_number}\tSize:{len(item_ids)}")
        print(f"\tAvg viewees:\t{int(view_sum)}")
        print(f"\tAvg savees:\t{int(save_sum)}")
        print(f"\tAvg buyees:\t{int(buy_sum)}")

def general_predictability_behaviour_of_users():
    cache_dir = os.path.join(curr_dir, "../cache")
    try:
        cache_file_users = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_users.pkl")
        cache_file_items = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_items.pkl")
    except:
        print("The desired files don't seem to exist.")
        return

    with open(cache_file_users, 'rb') as f:
        user_clusters = pickle.load(f)
    with open(cache_file_items, 'rb') as f:
        item_clusters = pickle.load(f)
    
    user_clusters_sorted = sorted(user_clusters, key=lambda x: len(x), reverse=True)
    item_clusters_sorted = sorted(item_clusters, key=lambda x: len(x))

    idx_to_user_id = list(G.user_to_items.keys())
    idx_to_item_id = list(G.item_to_users.keys())
    
    # Enable lookup of labels
    item_id_to_label = {}
    for cluster_number, idx in enumerate(item_clusters_sorted):
        for i in idx:
            item_id_to_label[idx_to_item_id[int(i)]] = cluster_number

    # Filter user clusters by size 
    k = 10
    user_clusters_filtered = user_clusters_sorted[:k]

    # Create subplots layout
    ncols = 5
    nrows = (k + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    for cluster_number, idx in enumerate(user_clusters_filtered): 
        count_view = np.zeros(len(item_clusters))
        count_save = np.zeros(len(item_clusters))
        count_buy = np.zeros(len(item_clusters))
        user_ids = [idx_to_user_id[i] for i in idx]
        
        for user_id in user_ids:
            for item_id in G.user_to_items_view[user_id]:
                item_cluster_number = item_id_to_label[item_id]
                count_view[item_cluster_number] += 1 / len(item_clusters_sorted[item_cluster_number])    
            for item_id in G.user_to_items_save[user_id]:
                item_cluster_number = item_id_to_label[item_id]
                count_save[item_cluster_number] += 1 / len(item_clusters_sorted[item_cluster_number])    
            for item_id in G.user_to_items_buy[user_id]:
                item_cluster_number = item_id_to_label[item_id]
                count_buy[item_cluster_number] += 1 / len(item_clusters_sorted[item_cluster_number])    
        count_view /= np.sum(count_view)
        count_save /= np.sum(count_save)
        count_buy /= np.sum(count_buy)

        # Plot in the corresponding subplot
        ax = axes[cluster_number]
        ax.bar(range(len(count_buy)), count_view)
        ax.bar(range(len(count_buy)), count_save)
        ax.bar(range(len(count_buy)), count_buy)
        ax.set_xlabel('Item Cluster')
        ax.set_ylabel('Normalized Frequency')
        ax.set_title(f'User Cluster {cluster_number + 1} (size: {len(idx)})')
        
    # Hide any unused subplots
    for i in range(k, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def general_predictability_behaviour_of_items():
    cache_dir = os.path.join(curr_dir, "../cache")
    try:
        cache_file_users = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_users.pkl")
        cache_file_items = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_items.pkl")
    except:
        print("The desired files don't seem to exist.")
        return

    with open(cache_file_users, 'rb') as f:
        user_clusters = pickle.load(f)
    with open(cache_file_items, 'rb') as f:
        item_clusters = pickle.load(f)
    
    user_clusters_sorted = sorted(user_clusters, key=lambda x: len(x), reverse=True)
    item_clusters_sorted = sorted(item_clusters, key=lambda x: len(x), reverse=True)

    idx_to_user_id = list(G.user_to_items.keys())
    idx_to_item_id = list(G.item_to_users.keys())
    
    # Enable lookup of user cluster labels
    user_id_to_label = {}
    for cluster_number, idx in enumerate(user_clusters_sorted):
        for i in idx:
            user_id_to_label[idx_to_user_id[int(i)]] = cluster_number

    # Filter item clusters by size 
    k = 10
    item_clusters_filtered = item_clusters_sorted[:k]

    # Create subplots layout
    ncols = 5
    nrows = (k + ncols - 1) // ncols  # Ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    for cluster_number, idx in enumerate(item_clusters_filtered): 
        count_view = np.zeros(len(user_clusters))
        count_save = np.zeros(len(user_clusters))
        count_buy = np.zeros(len(user_clusters))
        item_ids = [idx_to_item_id[i] for i in idx]
        
        for item_id in item_ids:
            # Check which users viewed this item
            for user_id in G.item_to_users_view.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                count_view[user_cluster_number] += 1 / len(user_clusters_sorted[user_cluster_number])    
            # Check which users saved this item
            for user_id in G.item_to_users_save.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                count_save[user_cluster_number] += 1 / len(user_clusters_sorted[user_cluster_number])    
            # Check which users bought this item
            for user_id in G.item_to_users_buy.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                count_buy[user_cluster_number] += 1 / len(user_clusters_sorted[user_cluster_number])    
        
        # Normalize
        count_view /= np.sum(count_view) if np.sum(count_view) > 0 else 1
        count_save /= np.sum(count_save) if np.sum(count_save) > 0 else 1
        count_buy /= np.sum(count_buy) if np.sum(count_buy) > 0 else 1

        # Plot in the corresponding subplot
        ax = axes[cluster_number]
        ax.bar(range(len(count_buy)), count_view, label='View')
        ax.bar(range(len(count_buy)), count_save, label='Save')
        ax.bar(range(len(count_buy)), count_buy, label='Buy')
        ax.set_xlabel('User Cluster')
        ax.set_ylabel('Normalized Frequency')
        ax.set_title(f'Item Cluster {cluster_number + 1} (size: {len(idx)})')
        if cluster_number == 0:
            ax.legend()
        
    # Hide any unused subplots
    for i in range(k, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def general_predictability_behaviour_of_user_item_pairs():
    cache_dir = os.path.join(curr_dir, "../cache")
    try:
        cache_file_users = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_users.pkl") # in common sim
        cache_file_items = os.path.join(cache_dir, "recursive_spectral_clustering_clusters_items.pkl") # in common sim
        # cache_file_users = os.path.join(cache_dir, "rsc_all_interactions_users.pkl") # cosine
        # cache_file_items = os.path.join(cache_dir, "rsc_all_interactions_items.pkl") # cosine
    except:
        print("The desired files don't seem to exist.")
        return

    with open(cache_file_users, 'rb') as f:
        user_clusters = pickle.load(f)
    with open(cache_file_items, 'rb') as f:
        item_clusters = pickle.load(f)

    user_clusters_sorted = sorted(user_clusters, key=lambda x: sum([len(G.user_to_items[str(user_id+1)]) for user_id in list(x)]), reverse=True)
    item_clusters_sorted = sorted(item_clusters, key=lambda x: sum([len(G.item_to_users[str(item_id+1)]) for item_id in list(x)]), reverse=True)

    idx_to_user_id = list(G.user_to_items.keys())
    idx_to_item_id = list(G.item_to_users.keys())
    
    # Enable lookup of user cluster labels
    user_id_to_label = {}
    for cluster_number, idx in enumerate(user_clusters_sorted):
        for i in idx:
            user_id_to_label[idx_to_user_id[int(i)]] = cluster_number

    num_item_clusters = len(item_clusters)
    num_user_clusters = len(user_clusters)

    item_cluster_counts_view = np.zeros((num_item_clusters, num_user_clusters))
    item_cluster_counts_save = np.zeros((num_item_clusters, num_user_clusters))
    item_cluster_counts_buy = np.zeros((num_item_clusters, num_user_clusters))

    user_cluster_counts_view = np.zeros((num_user_clusters, num_item_clusters))
    user_cluster_counts_save = np.zeros((num_user_clusters, num_item_clusters))
    user_cluster_counts_buy = np.zeros((num_user_clusters, num_item_clusters))

    for cluster_number, idx in enumerate(item_clusters_sorted): 
        item_ids = [idx_to_item_id[i] for i in idx]
        item_cluster_size = len(item_clusters[cluster_number])
        for item_id in item_ids:
            # Check which users viewed this item
            for user_id in G.item_to_users_view.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                user_cluster_size = len(user_clusters_sorted[user_cluster_number]) 
                item_cluster_counts_view[cluster_number, user_cluster_number] += 1 / (user_cluster_size * item_cluster_size)  
            # Check which users saved this item
            for user_id in G.item_to_users_save.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                user_cluster_size = len(user_clusters_sorted[user_cluster_number]) 
                item_cluster_counts_save[cluster_number, user_cluster_number] += 1 / (user_cluster_size * item_cluster_size)      
            # Check which users bought this item
            for user_id in G.item_to_users_buy.get(item_id, []):
                user_cluster_number = user_id_to_label[user_id]
                user_cluster_size = len(user_clusters_sorted[user_cluster_number]) 
                item_cluster_counts_buy[cluster_number, user_cluster_number] += 1 / (user_cluster_size * item_cluster_size)     
    
    # Normalize rows
    item_cluster_counts_view /= np.sum(item_cluster_counts_view, axis=1, keepdims=True)
    item_cluster_counts_save /= np.sum(item_cluster_counts_save, axis=1, keepdims=True)
    item_cluster_counts_buy /= np.sum(item_cluster_counts_buy, axis=1, keepdims=True)
    
    # Enable lookup of item cluster labels
    item_id_to_label = {}
    for cluster_number, idx in enumerate(item_clusters_sorted):
        for i in idx:
            item_id_to_label[idx_to_item_id[int(i)]] = cluster_number
    
    for cluster_number, idx in enumerate(user_clusters_sorted): 
        user_ids = [idx_to_user_id[i] for i in idx]
        user_cluster_size = len(user_clusters_sorted[cluster_number])
        for user_id in user_ids:
            # Check which items this user viewed
            for item_id in G.user_to_items_view.get(user_id, []):
                item_cluster_number = item_id_to_label[item_id]
                item_cluster_size = len(item_clusters_sorted[item_cluster_number])
                user_cluster_counts_view[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
            # Check which items this user saved
            for item_id in G.user_to_items_save.get(user_id, []):
                item_cluster_number = item_id_to_label[item_id]
                item_cluster_size = len(item_clusters_sorted[item_cluster_number])
                user_cluster_counts_save[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
                user_cluster_counts_view[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
            # Check which items this user bought
            for item_id in G.user_to_items_buy.get(user_id, []):
                item_cluster_number = item_id_to_label[item_id]
                item_cluster_size = len(item_clusters_sorted[item_cluster_number])
                user_cluster_counts_buy[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
                user_cluster_counts_view[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
                user_cluster_counts_save[cluster_number, item_cluster_number] += 1 / (item_cluster_size )#* user_cluster_size)
    
    # Normalize rows
    user_cluster_counts_view /= np.sum(user_cluster_counts_view, axis=1, keepdims=True)
    user_cluster_counts_save /= np.sum(user_cluster_counts_save, axis=1, keepdims=True)
    user_cluster_counts_buy /= np.sum(user_cluster_counts_buy, axis=1, keepdims=True)

    item_user_clusters_buy_prob = user_cluster_counts_buy  
    item_user_clusters_save_prob = user_cluster_counts_save 
    item_user_clusters_view_prob = user_cluster_counts_view 

    cache_dir = os.path.join(curr_dir, "../cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "probability_distr_cumulative.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump([user_id_to_label, 
                    item_id_to_label,
                    item_clusters_sorted, 
                    user_clusters_sorted,
                    item_user_clusters_view_prob, 
                    item_user_clusters_save_prob, 
                    item_user_clusters_buy_prob
                    ], f)
    print(f"Clusters saved to {cache_file}")

    plot_as_surfs(item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob)

def plot_as_surfs(item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob):

    # Create meshgrid for coordinates
    num_users, num_items = item_user_clusters_buy_prob.shape
    X = np.arange(num_items)
    Y = np.arange(num_users)
    X, Y = np.meshgrid(X, Y)
    
    # Buy probability surface - separate figure
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X, Y, item_user_clusters_buy_prob, cmap='hot', 
                              edgecolor='none', alpha=0.9)
    ax1.set_title('Item-User Cluster Buy Probability', fontsize=14, pad=20)
    ax1.set_xlabel('Item Clusters', fontsize=12)
    ax1.set_ylabel('User Clusters', fontsize=12)
    ax1.set_zlabel('Probability', fontsize=12)
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Save probability surface - separate figure
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(X, Y, item_user_clusters_save_prob, cmap='hot', 
                              edgecolor='none', alpha=0.9)
    ax2.set_title('Item-User Cluster Save Probability', fontsize=14, pad=20)
    ax2.set_xlabel('Item Clusters', fontsize=12)
    ax2.set_ylabel('User Clusters', fontsize=12)
    ax2.set_zlabel('Probability', fontsize=12)
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    # View probability surface - separate figure
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(X, Y, item_user_clusters_view_prob, cmap='hot', 
                              edgecolor='none', alpha=0.9)
    ax3.set_title('Item-User Cluster View Probability', fontsize=14, pad=20)
    ax3.set_xlabel('Item Clusters', fontsize=12)
    ax3.set_ylabel('User Clusters', fontsize=12)
    ax3.set_zlabel('Probability', fontsize=12)
    fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    # View probability surface - separate figure
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    surf4 = ax4.plot_surface(X, Y, item_user_clusters_view_prob + item_user_clusters_save_prob + item_user_clusters_buy_prob, cmap='hot', 
                              edgecolor='none', alpha=0.9)
    ax4.set_title('Sum of probabilities', fontsize=14, pad=20)
    ax4.set_xlabel('Item Clusters', fontsize=12)
    ax4.set_ylabel('User Clusters', fontsize=12)
    ax4.set_zlabel('Probability', fontsize=12)
    ax4.set_zlim(0,1)
    fig4.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
    
    plt.show()

def verify_distribution_properties():
    pass

def plot_as_imgs(item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob):

    # display the matricies as images 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Buy probability matrix
    im1 = axes[0].imshow(item_user_clusters_buy_prob, cmap='hot', interpolation='nearest', aspect='auto')
    axes[0].set_title('Item-User Cluster Buy Probability')
    axes[0].set_xlabel('Item Clusters')
    axes[0].set_ylabel('User Clusters')
    plt.colorbar(im1, ax=axes[0])
    
    # Save probability matrix
    im2 = axes[1].imshow(item_user_clusters_save_prob, cmap='hot', interpolation='nearest', aspect='auto')
    axes[1].set_title('Item-User Cluster Save Probability')
    axes[1].set_xlabel('Item Clusters')
    axes[1].set_ylabel('User Clusters')
    plt.colorbar(im2, ax=axes[1])
    
    # View probability matrix
    im3 = axes[2].imshow(item_user_clusters_view_prob, cmap='hot', interpolation='nearest', aspect='auto')
    axes[2].set_title('Item-User Cluster View Probability')
    axes[2].set_xlabel('Item Clusters')
    axes[2].set_ylabel('User Clusters')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

def load_model(model_filename="probability_distr_cosine.pkl"):
    cache_dir = os.path.join(curr_dir, "../cache")
    cache_file = os.path.join(cache_dir, model_filename)
    with open(cache_file, 'rb') as f:
        user_id_to_label, item_id_to_label, item_clusters_sorted, user_clusters_sorted, item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob = pickle.load(f)

    return user_id_to_label, item_id_to_label,item_clusters_sorted, user_clusters_sorted,item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob

def predict(user_id, item_id, model):
    user_id_to_label, item_id_to_label, item_clusters_sorted, user_clusters_sorted, item_user_clusters_view_prob, item_user_clusters_save_prob, item_user_clusters_buy_prob = model

    user_cluster_number = user_id_to_label[user_id]
    item_cluster_number = item_id_to_label[item_id]

    prob_of_view = item_user_clusters_view_prob[user_cluster_number, item_cluster_number]
    prob_of_save = item_user_clusters_save_prob[user_cluster_number, item_cluster_number]
    prob_of_buy = item_user_clusters_buy_prob[user_cluster_number, item_cluster_number]
    
    prob_of_interaction = (prob_of_view + prob_of_save + prob_of_buy)

    prob_of_no_interaction = (1 - prob_of_interaction)

    item_cluster = set([str(x) for x in item_clusters_sorted[item_cluster_number]])
    num_bought = len(G.user_to_items_buy[user_id])
    num_saved = len(G.user_to_items_save[user_id])
    num_viewed = len(G.user_to_items_view[user_id])
    total = len(G.user_to_items[user_id])
    # print([num_viewed, num_saved, num_bought, len(item_cluster)])
    prob_buy = num_bought / total * prob_of_interaction
    prob_save = num_saved  / total * prob_of_interaction
    prob_view = num_viewed / total * prob_of_interaction
    prob_no_interaction = (1 - (prob_buy + prob_view + prob_save)) * prob_of_no_interaction 

    probs = [prob_no_interaction, prob_view, prob_save, prob_view]
    
    # print(int(100 * np.sum(probs)), [int(100 *x) for x in probs])

    # pred = np.argmax(probs)
    # Sample from the probability distribution instead of taking argmax
    probs_normalized = np.array(probs) / np.sum(probs)  # Normalize to ensure sum = 1.0
    pred = np.random.choice(4, p=probs_normalized)
    pred_vec = np.zeros(4)
    pred_vec[int(pred)] = 1
    return pred_vec

def evaluate_on_training_set():
    num_users = len(G.user_to_items.keys())
    num_items = len(G.item_to_users.keys())

    model = load_model("probability_distr.pkl")

    # Initialize per-class metrics for F1-macro calculation
    TP = np.zeros(4)  # True Positives for each class
    FP = np.zeros(4)  # False Positives for each class
    FN = np.zeros(4)  # False Negatives for each class

    for i in tqdm(range(10000)):
        # sample user 
        user_id = G.user_to_items[int(np.random.choice(num_users))]
        # sample item
        item_id = G.item_to_users[int(np.random.choice(num_items))]
        # lookup interaction
        true_label = 0
        if (user_id, item_id) in G.user_item_to_interaction:
            true_label = int(G.user_item_to_interaction[(user_id, item_id)])

        prediction = predict(user_id, item_id, model)  # Returns one-hot vector
        predicted_label = np.argmax(prediction)
        
        # Update per-class metrics
        for i in range(4):
            if true_label == i and predicted_label == i:
                TP[i] += 1
            elif true_label != i and predicted_label == i:
                FP[i] += 1
            elif true_label == i and predicted_label != i:
                FN[i] += 1

     # Compute per-class precision, recall, and F1 score
    precision = TP / (TP + FP + 1e-10)  # Add small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    F1_score = 2 * precision * recall / (precision + recall + 1e-10)

    # Print detailed metrics
    print("Per-class metrics:")
    for i in range(4):
        print(f"Class {i}: TP={TP[i]:.0f}, FP={FP[i]:.0f}, FN={FN[i]:.0f}, "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={F1_score[i]:.4f}")
    
    # Macro-F1 is the average of per-class F1 scores
    Macro_F1_score = np.mean(F1_score)

    print(f"\nMacro F1 Score: {Macro_F1_score:.4f}")

def evaluate_on_data(filename):
    data = []
    with open(filename, 'r') as f: 
        for line in tqdm(f.readlines()):
            # Get new sample
            user_id, item_id, interaction = tuple(line.split("\t"))
            data.append((user_id, item_id, interaction))

    model = load_model("probability_distr.pkl")
    user_id_to_label = model[0]
    user_clusters_sorted = model[3]

    # Initialize per-class metrics for F1-macro calculation
    TP = np.zeros(4)  # True Positives for each class
    FP = np.zeros(4)  # False Positives for each class
    FN = np.zeros(4)  # False Negatives for each class

    for (user_id, item_id, interaction) in tqdm(data):
        # Get true label and prediction
        true_label = int(interaction)
        prediction = predict(user_id, item_id, model)  # Returns one-hot vector
        predicted_label = np.argmax(prediction)
        
        # Update per-class metrics
        for i in range(4):
            if true_label == i and predicted_label == i:
                TP[i] += 1
            elif true_label != i and predicted_label == i:
                FP[i] += 1
            elif true_label == i and predicted_label != i:
                FN[i] += 1

    # Compute per-class precision, recall, and F1 score
    precision = TP / (TP + FP + 1e-10)  # Add small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-10)
    F1_score = 2 * precision * recall / (precision + recall + 1e-10)

    # Print detailed metrics
    print("Per-class metrics:")
    for i in range(4):
        print(f"Class {i}: TP={TP[i]:.0f}, FP={FP[i]:.0f}, FN={FN[i]:.0f}, "
              f"Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={F1_score[i]:.4f}")
    
    # Macro-F1 is the average of per-class F1 scores
    Macro_F1_score = np.mean(F1_score)

    print(f"\nMacro F1 Score: {Macro_F1_score:.4f}")

if __name__=='__main__':
    general_predictability_behaviour_of_user_item_pairs()
    # test_RSC_on_items()
    # test_RSC_for_intersection_count_sim_on_users()
    # general_predictability_behaviour_of_user_item_pairs()
    
    # # Evaluation
    # path = os.path.join(curr_dir, "../../../data/task1_train.tsv")
    # # path = os.path.join(curr_dir, "../../../data/task1_val_answers.tsv")
    # evaluate_on_data(path)

    #evaluate_on_training_set()