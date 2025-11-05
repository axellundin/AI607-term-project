from settings import *
from util.graphs import *

training_data_file = "task1_train.tsv"
G = InteractionGraph(training_data_file)

adj = G.get_intersection_similarity_matrix(G.user_to_items, filename=users_intersection_similarity_adj_matrix)
normalized = normalize_weighted_adj_matrix(adj, users_intersection_similarity_adj_matrix_normalized)

del adj, normalized

adj = G.get_intersection_similarity_matrix(G.item_to_users, filename=items_intersection_similarity_adj_matrix)
normalized = normalize_weighted_adj_matrix(adj, items_intersection_similarity_adj_matrix_normalized)

