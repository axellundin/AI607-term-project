import numpy as np
from scipy.sparse import coo_matrix 

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
        
        for i, (node_i, conn_i) in enumerate(target_items):
            conn_i_set = set(conn_i)
            for j, (node_j, conn_j) in enumerate(target_items[i:], start=i):
                conn_j_set = set(conn_j)
                shared_count = len(conn_i_set.intersection(conn_j_set))
                val = int(shared_count >= k)
                X.append(i)
                Y.append(j)
                data.append(val)
                if i != j: # Symmetric
                    X.append(j)
                    Y.append(i)
                    data.append(val)
        
        coo = coo_matrix((data, (X, Y)), shape=(num_nodes, num_nodes))
        return coo 


