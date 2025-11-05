from util.graphs import * 
import matplotlib.pyplot as plt
from scipy.sparse import csr_array 


training_data_file = "task1_train.tsv"
G = InteractionGraph(training_data_file)

adj = G.get_intersection_similarity_matrix(G.user_to_items, filename=users_intersection_similarity_adj_matrix)
normalized = normalize_weighted_adj_matrix(adj, users_intersection_similarity_adj_matrix_normalized)

threshold = 0.0002
normalized[normalized < threshold] = 0

sparse_normalized = csr_array(normalized) 

print("staring constructing graph")
user_graph = nx.Graph(normalized)

print("starting pagerank")
rank = nx.pagerank(user_graph)
print("done with pagerank")

