from settings import * 
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

class InteractionGraph: 
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

        self.cache_location = cache_dir
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
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as f: 
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

    def get_intersection_similarity_matrix(self, target:dict, use_cache=True, filename="intersection_similarity"):
        if use_cache:
            try: 
                print("Checking cache")
                return self.load_cache(filename)
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

def normalize_weighted_adj_matrix(adj, filename, use_cache=True) -> np.ndarray:
    file_path = os.path.join(cache_dir, filename)
    if use_cache:
            try: 
                print("Checking cache")
                return np.load(file_path) 
            except Exception as e:
                print("No cache exists, computing..")
        
    new_adj = np.zeros(adj.shape)
    N = adj.shape[0]
    for idx1 in tqdm(range(N)):
        total_similarity = np.sum(adj[idx1,:])
        for idx2 in range(N):
            if idx1 == idx2: 
                continue 
            new_adj[idx1,idx2] = adj[idx1, idx2] / total_similarity

    with open(file_path, "wb") as f: 
        np.save(f, new_adj)

    return new_adj

def get_users_graph_from_adj_matrix(G: InteractionGraph, use_cache=True) -> nx.Graph:
    adj = G.get_intersection_similarity_matrix(G.user_to_items, filename=users_intersection_similarity_adj_matrix)

    user_graph = nx.DiGraph()

    for idx1, user_id_1 in enumerate(G.user_to_items.keys()):
        total_similarity = np.sum(adj[idx1,:])
        for idx2, user_id_2 in enumerate(G.user_to_items.keys()):
            if user_id_1 == user_id_2: 
                continue 
            directed_normalized_similarity = adj[idx1, idx2] / total_similarity
            user_graph.add_edge(user_id_1, 
                                user_id_2, 
                                weight=directed_normalized_similarity)

    return user_graph

def get_items_graph_from_adj_matrix(G: InteractionGraph, use_cache=True) -> nx.Graph:
    adj = G.get_intersection_similarity_matrix(G.item_to_users, filename=items_intersection_similarity_adj_matrix)

    item_graph = nx.DiGraph()

    for idx1, item_id_1 in enumerate(G.item_to_users.keys()):
        total_similarity = np.sum(adj[idx1,:])
        for idx2, item_id_2 in enumerate(G.item_to_users.keys()):
            if item_id_1 == item_id_2: 
                continue 
            directed_normalized_similarity = adj[idx1, idx2] / total_similarity
            item_graph.add_edge(item_id_1, 
                                item_id_2, 
                                weight=directed_normalized_similarity)

    return item_graph

def check_sparsity(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    nonzero_elements = np.count_nonzero(matrix)
    sparsity = 1 - (nonzero_elements / total_elements)
    density = nonzero_elements / total_elements

    print(f"Matrix shape: {matrix.shape}")
    print(f"Total possible edges: {total_elements:,}")
    print(f"Actual edges (non-zero): {nonzero_elements:,}")
    print(f"Sparsity: {sparsity:.4%}")  
    print(f"Density: {density:.4%}")  