import os 

# directories
src_dir = os.path.dirname(__file__)
data_dir = os.path.join(src_dir, "../data")
cache_dir = os.path.join(src_dir, "cache")
results_dir = os.path.join(src_dir, "results")
models_dir = os.path.join(results_dir, "models")
training_data_filename = "task1_train.tsv"
val_data_filename = "task1_val_answers.tsv"

# Cache filenames
users_intersection_similarity_adj_matrix = "users_sim_adj_matrix.npy"
items_intersection_similarity_adj_matrix = "items_sim_adj_matrix.npy"
users_intersection_similarity_adj_matrix_normalized = "users_sim_adj_matrix_normalized.npy"
items_intersection_similarity_adj_matrix_normalized = "items_sim_adj_matrix_normalized.npy"