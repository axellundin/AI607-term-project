import os 

# directories
src_dir = os.path.dirname(__file__)
data_dir = os.path.join(src_dir, "../data")
cache_dir = os.path.join(src_dir, "cache")
results_dir = os.path.join(src_dir, "results")
models_dir = os.path.join(results_dir, "models")
models_task2_dir = os.path.join(results_dir, "models_task2")
training_data_filename = "task1_train.tsv"
val_data_filename = "task1_val_answers.tsv"
test_data_filename = "task1_test_queries.tsv"