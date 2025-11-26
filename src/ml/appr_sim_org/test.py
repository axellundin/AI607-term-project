import torch
import os
import argparse
from settings import *
from ml.appr_sim_org.data import HeteroGraphDataLoader
from ml.appr_sim_org.models import create_model
from ml.appr_sim_org.evaluation import Evaluator

def print_model_info(checkpoint):
    print("\n" + "=" * 40)
    print("MODEL INFORMATION")
    print("=" * 40)
    
    if 'epoch' in checkpoint:
        print(f"Training Epochs: {checkpoint['epoch']}")
    
    if 'hyperparameters' in checkpoint:
        print("\nHyperparameters:")
        hp = checkpoint['hyperparameters']
        for key, value in hp.items():
            print(f"  {key}: {value}")
            
    # Basic stats
    if 'user2idx' in checkpoint:
        print(f"\nUsers in mapping: {len(checkpoint['user2idx'])}")
    if 'item2idx' in checkpoint:
        print(f"Items in mapping: {len(checkpoint['item2idx'])}")
        
    print("=" * 40 + "\n")

def run_test(model_name=None):
    # If no model name provided, try to find a default or use command line
    if model_name is None:
        # Default fallback or try to parse args if running as script
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('model_name', nargs='?', help='Name of the model file')
            args = parser.parse_args()
            if args.model_name:
                model_name = args.model_name
            else:
                # Try to find a model in models_dir
                potential_models = [f for f in os.listdir(models_dir) if f.endswith('.pt') and 'hetero_model' in f]
                if potential_models:
                    # Sort by modification time
                    potential_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                    model_name = potential_models[0]
                    print(f"No model specified, using latest found: {model_name}")
                else:
                    print("No model specified and no default models found.")
                    return
        except:
             # Fallback if argparse fails (e.g. when called from another script)
             pass

    device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = os.path.join(models_dir, model_name)
    if not model_path.endswith('.pt'):
        model_path += '.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    print_model_info(checkpoint)
    
    hp = checkpoint['hyperparameters']

    # Reconstruct model
    model = create_model(
        hp.get('model_type', 'SAGE'), 
        hp['num_users'], 
        hp['num_items'], 
        hp['embedding_dim'],
        hp['hidden_channels'], 
        hp.get('num_layers', 2), 
        hp.get('dropout', 0.3), 
        hp.get('num_classes', 4),
        aggr=hp.get('aggr', 'mean')
    ).to(device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"Failed to load state dict: {e}")
        return

    model.eval()

    # Load data
    data_loader = HeteroGraphDataLoader(data_dir) 
    print("Loading data...")
    dataset_name = "task1"
    
    # Load graph structure for message passing
    data, _, _, _ = data_loader.load_dataset(f"{dataset_name}_train.tsv")
    data = data.to(device)

    # Load validation data
    val_data = data_loader.load_validation_dataset(f"{dataset_name}_val_answers.tsv")
    val_pairs = list(val_data.keys())
    val_labels = [val_data[pair] for pair in val_pairs]

    # Evaluate
    evaluator = Evaluator(model, data, device, checkpoint['user2idx'], checkpoint['item2idx'])
    metrics = evaluator.evaluate(val_pairs, val_labels)
    evaluator.print_results(metrics)

if __name__ == "__main__":
    run_test()

