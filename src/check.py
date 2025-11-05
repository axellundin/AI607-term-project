from settings import * 
from util.graphs import InteractionGraph

G = InteractionGraph(os.path.join(data_dir, training_data_filename))

with open(os.path.join(data_dir, val_data_filename), 'r') as f:
    count = 0
    for entry in f.readlines():
        user_id, item_id, interaction = entry.split("\t")
        if (user_id, item_id) in G.user_item_to_interaction.keys():
            count += 1

print(count)