import numpy as np
from scipy.sparse import coo_matrix 
from tqdm import tqdm
import os
import torch
from torch_geometric.data import HeteroData
import random
import numpy as np

class FullGraph: 
    def __init__(self, filename:str=None, 
                 filename_view:str=None, 
                 filename_save:str=None, 
                 filename_buy:str=None):
        self.user_to_items = {}
        self.item_to_users = {}
        self.user_item_to_interaction = {}
        # Interaction filtered graphs 
        self.user_to_items_view = {}
        self.user_to_items_save = {}
        self.user_to_items_buy  = {}

        self.item_to_users_view = {}
        self.item_to_users_save = {}
        self.item_to_users_buy  = {}
        
        # Store filename references
        self.filename = filename
        self.filename_view = filename_view
        self.filename_save = filename_save
        self.filename_buy = filename_buy

        self.cache_location = os.path.join(os.path.dirname(__file__), "../cache")
        if not os.path.exists(self.cache_location):
            os.mkdir(self.cache_location)
        # Load data
        if filename:
            self.load_data(filename)
        if filename_view:
            self.load_data(filename_view)
        if filename_save:
            self.load_data(filename_save)
        if filename_buy:
            self.load_data(filename_buy)
    
    def load_data(self, filename:str):
        with open(filename, 'r') as f: 
            for line in f.readlines():
                user_id, item_id, interaction = tuple(line.split("\t"))
                interaction = int(interaction)
                # Initialize new entries
                if user_id not in self.user_to_items.keys():
                    self.user_to_items[user_id] = []
                    self.user_to_items_view[user_id] = [] 
                    self.user_to_items_save[user_id] = [] 
                    self.user_to_items_buy[user_id] = []  
                if item_id not in self.item_to_users.keys():
                    self.item_to_users[item_id] = []
                    self.item_to_users_view[item_id] = []
                    self.item_to_users_save[item_id] = []
                    self.item_to_users_buy[item_id] = []
                # Add the edges
                self.user_to_items[user_id] += [item_id]
                self.item_to_users[item_id] += [user_id]
                
                if interaction == 1:
                    self.user_to_items_view[user_id] += [item_id] 
                    self.item_to_users_view[item_id] += [user_id]
                elif interaction == 2:
                    self.user_to_items_save[user_id] += [item_id] 
                    self.item_to_users_save[item_id] += [user_id]
                elif interaction == 3:
                    self.user_to_items_buy[user_id] += [item_id] 
                    self.item_to_users_buy[item_id] += [user_id]

                self.user_item_to_interaction[(user_id,item_id)] = interaction

    def load_cache(self, filename):
        return np.load(os.path.join(self.cache_location, filename))

    def build_pyg_graph(self, num_negative_samples_per_pos=2):
        """
        Build PyG graph with proper separation of structure and prediction edges.
        
        Args:
            num_negative_samples_per_pos: Ratio of negative to positive samples.
                                          Set to 0 for evaluation (no negative sampling).
        
        Returns:
            data: HeteroData object with:
                - edge_index: Only positive edges for message passing (graph structure)
                - edge_label_index: All edges (pos + neg) for prediction
                - edge_label: Labels for all prediction edges
            user2idx: User ID to index mapping
            item2idx: Item ID to index mapping
        """
        data = HeteroData()

        # Map string IDs → integer indices
        users = list(self.user_to_items.keys())
        items = list(self.item_to_users.keys())
        user2idx = {u: i for i, u in enumerate(users)}
        item2idx = {it: i for i, it in enumerate(items)}

        # Collect positive edges
        src, dst, labels = [], [], []
        for (u, i), inter in self.user_item_to_interaction.items():
            src.append(user2idx[u])
            dst.append(item2idx[i])
            labels.append(inter)  # 1=view, 2=save, 3=buy

        # Negative sampling (no_interaction = 0)
        num_pos = len(src)
        num_neg = num_pos * num_negative_samples_per_pos
        neg_src, neg_dst = [], []

        if num_neg > 0:
            sampled = set()
            attempts = 0
            max_attempts = num_neg * 10
            
            while len(neg_src) < num_neg and attempts < max_attempts:
                u = random.choice(users)
                i = random.choice(items)
                if (u, i) not in self.user_item_to_interaction and (u, i) not in sampled:
                    neg_src.append(user2idx[u])
                    neg_dst.append(item2idx[i])
                    sampled.add((u, i))
                attempts += 1

        # Build hetero graph
        data['user'].num_nodes = len(users)
        data['item'].num_nodes = len(items)
        
        # CRITICAL: edge_index contains ONLY positive edges for message passing
        data['user', 'interacts', 'item'].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )
        
        # edge_label_index contains ALL edges (pos + neg) for prediction
        if num_neg > 0:
            all_src = src + neg_src
            all_dst = dst + neg_dst
            all_labels = labels + [0] * len(neg_src)
        else:
            all_src = src
            all_dst = dst
            all_labels = labels
        
        data['user', 'interacts', 'item'].edge_label_index = torch.tensor(
            [all_src, all_dst], dtype=torch.long
        )
        data['user', 'interacts', 'item'].edge_label = torch.tensor(all_labels, dtype=torch.long)

        return data, user2idx, item2idx

    def compute_degree_vector(self, target: dict):
        deg = np.zeros(len(target.keys()), dtype=int)
        for i, A in enumerate(target.keys()):
            deg[i] = len(target[A])
        return deg

    def get_degree_distribution_vector(self, target: dict):
        deg = self.compute_degree_vector(target)
        max_deg = np.max(deg)
        deg_hist = np.zeros(max_deg+1, dtype=int)
        for A in deg: 
            deg_hist[A] += 1
        return deg_hist
    
    def compute_consumer_group_statistics_item_level_partition(self, item_id):
        buyers = self.item_to_users_buy[item_id] 
        partition = np.zeros((len(self.item_to_users),len(buyers))) # |I| x |Y(i)|
        
        # Create a mapping from item_id to index
        item_to_index = {item: idx for idx, item in enumerate(self.item_to_users.keys())}
        
        for i, user_id in enumerate(buyers): 
            bought_items = self.user_to_items_buy[user_id]
            # Convert item IDs to indices
            bought_items_indices = [item_to_index[item] for item in bought_items]
            partition[bought_items_indices, i] = 1 

        avg = np.mean(partition, axis=1)
        std = np.std(partition, axis=1)
        return avg, std, len(buyers)

    def get_k_in_common_adj_matrix(self, target:dict, k:int=1):
        num_nodes = len(target)
        X = []
        Y = []
        data = []
        target_items = list(target.items())
        min_shared = 10000
        for i, (node_i, conn_i) in enumerate(tqdm(target_items)):
            conn_i_set = set(conn_i)
            max_shared_count = 0
            for j, (node_j, conn_j) in enumerate(target_items[i:], start=i):
                if i == j: 
                    continue
                conn_j_set = set(conn_j)
                shared_count = len(conn_i_set.intersection(conn_j_set))
                if shared_count > max_shared_count: 
                    max_shared_count = shared_count
                val = int(shared_count >= k)
                X.append(i)
                Y.append(j)
                data.append(val)
                X.append(j)
                Y.append(i)
                data.append(val)
            if max_shared_count < min_shared: 
                min_shared = max_shared_count

        print(f"{min_shared=}")
        
        coo = coo_matrix((data, (X, Y)), shape=(num_nodes, num_nodes))
        return coo

    def get_cosine_similarity_matrix(self, target:dict, use_cache=True, filename="cosine_similarity"):
        if use_cache:
            try: 
                print("Checking cache")
                return self.load_cache(f"{filename}.npy")
            except Exception as e:
                print("No cache exists, computing..")
        num_nodes = len(target)
        adj = np.zeros((num_nodes, num_nodes))
        target_items = list(target.items())
        
        for i, (node_i, conn_i) in enumerate(tqdm(target_items)):
            conn_i_set = set(conn_i)
            n_neighbors_i = len(conn_i_set)
            for j, (node_j, conn_j) in enumerate(target_items[i:], start=i):
                if i==j:
                    continue
                conn_j_set = set(conn_j)
                n_neighbors_j = len(conn_j_set)
                shared_count = len(conn_i_set.intersection(conn_j_set))
                sqrt = np.sqrt(n_neighbors_i * n_neighbors_j)
                if sqrt == 0: 
                    continue
                val = shared_count / sqrt
                if val < 0.5: 
                    continue
                adj[i,j] = adj[j,i] = val
        
        if use_cache:
            file_path = os.path.join(self.cache_location, filename)
            np.save(file_path, adj)

        return adj

    def get_intersection_similarity_matrix(self, target:dict, use_cache=True, filename="intersection_count_similarity"):
        if use_cache:
            try: 
                print("Checking cache")
                return self.load_cache(f"{filename}.npy")
            except Exception as e:
                print("No cache exists, computing..")
        num_nodes = len(target.keys())
        print(f"{num_nodes=}")
        adj = np.zeros((num_nodes, num_nodes))
        target_items = list(target.items())
        
        for i, (node_i, conn_i) in enumerate(tqdm(target_items)):
            conn_i_set = set(conn_i)
            for j, (node_j, conn_j) in enumerate(target_items[i:], start=i):
                if i==j:
                    continue
                conn_j_set = set(conn_j)
                shared_count = len(conn_i_set.intersection(conn_j_set))
                adj[i,j] = adj[j,i] = shared_count
        
        if use_cache:
            file_path = os.path.join(self.cache_location, filename)
            np.save(file_path, adj)

        return adj