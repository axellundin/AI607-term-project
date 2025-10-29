import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
import pickle
import os
from typing import Dict, List, Tuple, Any

def spectral_clustering(
    entity_graph: Dict[str, List[str]],
    n_clusters: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """
    Perform spectral clustering on entities (users or items) based on their connections.
    
    Args:
        entity_graph: Dictionary mapping entities to their connections (e.g., users to items or items to users)
        n_clusters: Number of clusters to find
        random_state: Random state for reproducibility
        
    Returns:
        labels: Cluster labels for each entity
        entity_list: List of entities in order corresponding to labels
    """
    # Create entity-to-index mapping
    entity_list = list(entity_graph.keys())
    entity_to_index = {entity: idx for idx, entity in enumerate(entity_list)}
    n_entities = len(entity_list)
    
    # Get all unique connections
    all_connections = set()
    for connections in entity_graph.values():
        all_connections.update(connections)
    all_connections = sorted(list(all_connections))
    connection_to_index = {conn: idx for idx, conn in enumerate(all_connections)}
    n_connections = len(all_connections)
    
    print(f"Building adjacency matrix for {n_entities} entities with {n_connections} unique connections...")
    
    # Build binary feature matrix (entities x connections) using sparse matrix
    # This is much faster and memory efficient
    row_indices = []
    col_indices = []
    
    for entity_idx, entity in enumerate(entity_list):
        connections = entity_graph[entity]
        for conn in connections:
            conn_idx = connection_to_index[conn]
            row_indices.append(entity_idx)
            col_indices.append(conn_idx)
    
    # Create sparse binary feature matrix
    feature_matrix = csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(n_entities, n_connections)
    )
    
    # Compute adjacency matrix: feature_matrix @ feature_matrix.T
    # Each element (i,j) = dot product = number of shared connections
    print("Computing adjacency matrix...")
    adjacency = feature_matrix.dot(feature_matrix.T)
    
    # Convert to dense array (or keep sparse if very large)
    adjacency = adjacency.toarray()
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(adjacency, 0)
    
    # Apply spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=random_state
    )
    
    print("Performing spectral clustering...")
    labels = clustering.fit_predict(adjacency)
    
    return labels, entity_list

def create_cluster_mappings(
    labels: np.ndarray,
    entity_list: List[str]
) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """
    Create bidirectional mappings between entities and clusters.
    
    Args:
        labels: Cluster labels for each entity
        entity_list: List of entities
        
    Returns:
        entity_to_cluster: Mapping from entity ID to cluster label
        cluster_to_entity: Mapping from cluster label to list of entity IDs
    """
    entity_to_cluster = {}
    cluster_to_entity = {}
    
    for entity_id, label in zip(entity_list, labels):
        # Initialize cluster list if needed
        if label not in cluster_to_entity:
            cluster_to_entity[label] = []
        
        entity_to_cluster[entity_id] = label
        cluster_to_entity[label].append(entity_id)
    
    return entity_to_cluster, cluster_to_entity


def get_clustering_cache_path(
    clustering_dir: str,
    entity_type: str,
    interaction_type: str,
    n_clusters: int
) -> str:
    """
    Generate a cache file path for clustering results.
    
    Args:
        clustering_dir: Directory to store cache files
        entity_type: 'user' or 'item'
        interaction_type: Type of interaction ('view', 'save', 'buy', or 'all')
        n_clusters: Number of clusters
        
    Returns:
        Path to cache file
    """
    return os.path.join(
        clustering_dir,
        f'{entity_type}_clustering_{interaction_type}_k{n_clusters}.pkl'
    )

def save_clustering_cache(
    entity_to_cluster: Dict[str, int],
    cluster_to_entity: Dict[int, List[str]],
    entity_type: str,
    interaction_type: str,
    n_clusters: int,
    clustering_dir: str = 'clustering_cache'
):
    """
    Save clustering results to pickle files.
    
    Args:
        entity_to_cluster: Mapping from entity ID to cluster label
        cluster_to_entity: Mapping from cluster label to list of entity IDs
        entity_type: 'user' or 'item'
        interaction_type: Type of interaction ('view', 'save', 'buy', or 'all')
        n_clusters: Number of clusters
        clustering_dir: Directory to save cache files
    """
    # Create directory if it doesn't exist
    os.chdir(os.path.join(os.path.dirname(__file__), "../"))
    os.makedirs(clustering_dir, exist_ok=True)
    
    # Generate cache file path
    cache_file = get_clustering_cache_path(
        clustering_dir=clustering_dir,
        entity_type=entity_type,
        interaction_type=interaction_type,
        n_clusters=n_clusters
    )
    
    # Save clustering data
    with open(cache_file, 'wb') as f:
        pickle.dump({
            f'{entity_type}_to_cluster': entity_to_cluster,
            f'cluster_to_{entity_type}': cluster_to_entity,
            'interaction_type': interaction_type,
            'n_clusters': n_clusters
        }, f)
    
    print(f"Saved {entity_type} clustering to {cache_file}")

def load_clustering_cache(
    entity_type: str,
    interaction_type: str,
    n_clusters: int,
    clustering_dir: str = 'clustering_cache'
) -> Tuple[Dict[str, int], Dict[int, List[str]], bool]:
    """
    Load clustering results from pickle files.
    
    Args:
        entity_type: 'user' or 'item'
        interaction_type: Type of interaction ('view', 'save', 'buy', or 'all')
        n_clusters: Number of clusters
        clustering_dir: Directory to load cache files from
        
    Returns:
        entity_to_cluster: Mapping from entity ID to cluster label
        cluster_to_entity: Mapping from cluster label to list of entity IDs
        success: Whether loading was successful
    """
    # Generate cache file path
    cache_file = get_clustering_cache_path(
        clustering_dir=clustering_dir,
        entity_type=entity_type,
        interaction_type=interaction_type,
        n_clusters=n_clusters
    )
    
    if not os.path.exists(cache_file):
        print(f"{entity_type.capitalize()} clustering cache not found at {cache_file}")
        return {}, {}, False
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            entity_to_cluster = data[f'{entity_type}_to_cluster']
            cluster_to_entity = data[f'cluster_to_{entity_type}']
        
        print(f"Loaded {entity_type} clustering from {cache_file}")
        return entity_to_cluster, cluster_to_entity, True
    
    except Exception as e:
        print(f"Error loading {entity_type} clustering cache: {e}")
        return {}, {}, False

def compute_embedding(
    user_id: str,
    item_id: str,
    cluster_to_user: Dict[int, List[str]],
    cluster_to_item: Dict[int, List[str]],
    user_to_items: Dict[str, List[str]],
    item_to_users: Dict[str, List[str]]
) -> np.ndarray:
    """
    Compute embedding for a user-item pair using clustering information.
    
    Args:
        user_id: User ID
        item_id: Item ID
        cluster_to_user: Mapping from cluster label to list of user IDs
        cluster_to_item: Mapping from cluster label to list of item IDs
        user_to_items: Mapping from user ID to list of item IDs
        item_to_users: Mapping from item ID to list of user IDs
    
    Returns:
        Concatenated embedding vector
    """
    user_embedding = np.zeros(len(cluster_to_user.keys()))
    
    buying_users_of_item = set(item_to_users[item_id])
    num_users = len(buying_users_of_item)
    for cluster, users in cluster_to_user.items():
        if num_users == 0: 
            break
        in_common = len(buying_users_of_item.intersection(users))
        user_embedding[cluster] = in_common / (num_users * len(users))

    item_embedding = np.zeros(len(cluster_to_item.keys()))

    items_bought_by_user = set(user_to_items[user_id])
    num_items = len(items_bought_by_user)
    for cluster, items in cluster_to_item.items():
        if num_items == 0: 
            break
        in_common = len(items_bought_by_user.intersection(items))
        item_embedding[cluster] = in_common / (num_items * len(items))

    return np.hstack((user_embedding, item_embedding))