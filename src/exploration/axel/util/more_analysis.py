from graphs import FullGraph 
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm 

curr_dir = os.path.dirname(__file__)

G = FullGraph(os.path.join(curr_dir, "../../../data/task1_train.tsv"))

def plot_interaction_behaviour():
    n_users = len(G.user_to_items)
    views = np.zeros(n_users)
    saves = np.zeros(n_users)
    buys = np.zeros(n_users)
    total_interactions = np.zeros(n_users)

    for i, user_id in enumerate(G.user_to_items.keys()):
        total = len(G.user_to_items[user_id])
        total_interactions[i] = total
        buys[i] = len(G.user_to_items_buy[user_id]) # / total if total > 0 else 0
        saves[i] = buys[i] + len(G.user_to_items_save[user_id]) # / total if total > 0 else 0
        views[i] = saves[i] + len(G.user_to_items_view[user_id]) # / total if total > 0 else 0

    # from mpl_toolkits.mplot3d import Axes3

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(views, saves, buys, c=total_interactions, cmap='viridis', alpha=0.5)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Total Interactions')

    ax.set_xlabel('Number of Views')
    ax.set_ylabel('Number of Saves')
    ax.set_zlabel('Number of Buys')
    ax.set_title('3D Scatter Plot of User Interactions (Views, Saves, Buys)\nColored by Total Interactions')

    plt.tight_layout()
    plt.show()

def get_user_trend_statistics():
    n_users = len(G.user_to_items)
    mean_popularity = np.zeros(n_users)
    in_common = np.zeros(n_users)
    
    print("Loading pre-computed intersection similarity matrix...")
    # Load the pre-computed intersection count matrix
    cache_file = os.path.join(curr_dir, "../cache/intersection_count_similarity_users.npy")
    intersection_matrix = np.load(cache_file)
    
    # Get user IDs in order (same order as the matrix)
    user_ids = list(G.user_to_items.keys())
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    
    print("Computing statistics...")
    for i, user_id in enumerate(tqdm(user_ids, desc="Processing users")):
        # Compute mean item popularity
        item_popularity = []
        for item_id in G.user_to_items_buy[user_id]:
            item_popularity.append(len(G.item_to_users_buy[item_id]))
        mean_popularity[i] = np.mean(item_popularity) if item_popularity else 0
        
        # Compute average similarity with neighbors using the pre-computed matrix
        # Neighbors are users who share at least one item (intersection_matrix[i, j] > 0)
        neighbor_intersections = intersection_matrix[i, :]
        neighbor_mask = neighbor_intersections > 0
        
        if neighbor_mask.sum() > 0:
            # Sum of all intersection counts divided by number of neighbors and number of items
            total_intersection = neighbor_intersections[neighbor_mask].sum()
            num_neighbors = neighbor_mask.sum()
            num_items = len(G.user_to_items[user_id])
            in_common[i] = total_intersection / num_neighbors / num_items if num_items > 0 else 0
        else:
            in_common[i] = 0

    in_common = (in_common - np.mean(in_common)) / np.std(in_common)
    mean_popularity = (mean_popularity - np.mean(mean_popularity)) /np.std(mean_popularity)

    # Single 2D scatter plot: mean vs neighbor similarity
    plt.figure(figsize=(8,6))
    plt.scatter(mean_popularity, in_common, alpha=0.6)
    plt.xlabel('Mean Item Popularity per User (standardized)')
    plt.ylabel('Average similarity with neighbors (standardized)')
    plt.title('User Trend: Mean item popularity vs neighbor similarity')
    plt.grid(True)
    plt.tight_layout()

    # 2D Histogram heatmap (density)
    plt.figure(figsize=(8,6))
    counts, xedges, yedges, im = plt.hist2d(
        mean_popularity,
        in_common,
        bins=40,
        cmap='viridis'
    )
    plt.colorbar(im, label='Number of Users')
    plt.xlabel('Mean Item Popularity per User (standardized)')
    plt.ylabel('Average similarity with neighbors (standardized)')
    plt.title('User Trend Density: Mean item popularity vs neighbor similarity (Heatmap)')
    plt.tight_layout()
    plt.savefig("heatmap_popularity.png")
    plt.show()

    # return mean_popularity, std_popularity
        

def get_item_turnover_rates():
    n_items = len(G.item_to_users)
    save_turnover_rate = np.zeros(n_items)
    buy_turnover_rate = np.zeros(n_items)

    for i, item_id in enumerate(G.item_to_users.keys()):
        view_interaction_count = len(G.item_to_users_view[item_id])
        save_interaction_count = len(G.item_to_users_save[item_id])
        buy_interaction_count = len(G.item_to_users_buy[item_id])
        save_turnover_rate[i] = save_interaction_count / (save_interaction_count + view_interaction_count)
        buy_turnover_rate[i] = buy_interaction_count / (save_interaction_count  + buy_interaction_count)

    return save_turnover_rate, buy_turnover_rate

def get_user_turnover_rates():
    n_users = len(G.user_to_items)
    save_turnover_rate = np.zeros(n_users)
    buy_turnover_rate = np.zeros(n_users)

    for i, user_id in enumerate(G.user_to_items.keys()):
        view_interaction_count = len(G.user_to_items_view[user_id])
        save_interaction_count = len(G.user_to_items_save[user_id])
        buy_interaction_count = len(G.user_to_items_buy[user_id])
        save_turnover_rate[i] = save_interaction_count / (save_interaction_count + view_interaction_count) 
        buy_turnover_rate[i] = buy_interaction_count / (save_interaction_count + buy_interaction_count)

    return save_turnover_rate, buy_turnover_rate

def plot_user_turnover_rates_scatter(STO, BTO):
    plt.figure(figsize=(8,6))
    plt.scatter(STO, BTO, alpha=0.6)
    plt.xlabel('Save turnover')
    plt.ylabel('Buy turnover')
    plt.title('Save vs Buy Turnover Rate (Scatter)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_user_turnover_rates_hist3d(STO, BTO, bins=30):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d)
    from matplotlib import cm

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute the 2D histogram
    hist, xedges, yedges = np.histogram2d(STO, BTO, bins=bins, range=[[0,1],[0,1]])

    # Build positions for the bars
    xpos, ypos = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2,
        indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dz = hist.ravel()

    dx = (xedges[1] - xedges[0]) * 0.8  # 80% of bin width for aesthetics
    dy = (yedges[1] - yedges[0]) * 0.8

    # Normalize counts for coloring
    norm = plt.Normalize(dz.min(), dz.max()) if dz.max() > 0 else plt.Normalize(0, 1)
    cmap = cm.viridis
    colors = cmap(norm(dz))

    bar_collection = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors, alpha=1)

    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.12)
    cbar.set_label('Count in bin')

    ax.set_xlabel('Save turnover')
    ax.set_ylabel('Buy turnover')
    ax.set_zlabel('Count in bin')
    ax.set_title('Save vs Buy Turnover Rate (3D Histogram)')
    ax.view_init(elev=30, azim=120)
    plt.tight_layout()
    plt.show()

def plot_user_turnover_rates_density_heatmap(STO, BTO, bandwidth=0.03, gridsize=100):
    from scipy.stats import gaussian_kde

    # Remove any possible NaNs for KDE
    mask = ~np.isnan(STO) & ~np.isnan(BTO)
    x = STO[mask]
    y = BTO[mask]

    # Perform kernel density estimate
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bandwidth)
    # Evaluate kde on meshgrid
    xi, yi = np.linspace(0, 1, gridsize), np.linspace(0, 1, gridsize)
    xi, yi = np.meshgrid(xi, yi)
    zi = kde(np.vstack([xi.ravel(), yi.ravel()]))

    plt.figure(figsize=(8,6))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='viridis')
    plt.xlabel('Save turnover')
    plt.ylabel('Buy turnover')
    plt.title('Item Save vs Buy Turnover Rate Heatmap')
    plt.colorbar(label="Density")
    plt.tight_layout()
    plt.savefig("heatmap_items.png")
    plt.show()

if __name__=='__main__':
    # STO, BTO = get_user_turnover_rates()
    # STO, BTO = get_item_turnover_rates()
    # # plot_user_turnover_rates_hist3d(STO, BTO)
    # plot_user_turnover_rates_density_heatmap(STO, BTO, gridsize=50)

    get_user_trend_statistics()



