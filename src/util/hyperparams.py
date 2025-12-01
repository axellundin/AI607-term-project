"""
Utility function to extract hyperparameters from PyTorch checkpoint files
and generate LaTeX table code.
"""
import torch
import os
from typing import Dict, Optional, Any


def print_hyperparams_latex(
    checkpoint_path: str,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    dropout_prob: Optional[float] = None,
    negative_sample_ratio: Optional[float] = None,
    num_epochs: Optional[int] = None,
    caption: str = "Hyperparameters used for training our GNN.",
    label: str = "tab:hyp"
) -> str:
    """
    Load a PyTorch checkpoint and generate LaTeX table code with hyperparameters.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        learning_rate: Learning rate (if not in checkpoint)
        batch_size: Batch size (if not in checkpoint)
        dropout_prob: Dropout probability (if not in checkpoint)
        negative_sample_ratio: Negative sample ratio (if not in checkpoint)
        num_epochs: Number of epochs (if not in checkpoint)
        caption: LaTeX table caption
        label: LaTeX table label
    
    Returns:
        LaTeX table code as a string
    """
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract hyperparameters from checkpoint
    hyperparams = checkpoint.get('hyperparameters', {})
    
    # Get embedding dimension
    embedding_dim = hyperparams.get('embedding_dim', None)
    if embedding_dim is None:
        raise ValueError("embedding_dim not found in checkpoint hyperparameters")
    
    # Get hidden channels
    hidden_channels = hyperparams.get('hidden_channels', None)
    if hidden_channels is None:
        raise ValueError("hidden_channels not found in checkpoint hyperparameters")
    
    # Try to get other hyperparameters from checkpoint or use provided values
    lr = hyperparams.get('learning_rate', learning_rate)
    bs = hyperparams.get('batch_size', batch_size)
    dropout = hyperparams.get('dropout_prob', hyperparams.get('dropout_edge_prob', dropout_prob))
    neg_ratio = hyperparams.get('negative_sample_ratio', negative_sample_ratio)
    # Only include epochs if explicitly provided (not in template)
    epochs = num_epochs if num_epochs is not None else None
    
    # Build LaTeX table
    latex_lines = [
        "\\begin{table}[H]",
        "    \\centering",
        "    \\begin{tabular}{l|c}",
        "        \\textbf{Parameter} & \\textbf{Value} \\\\",
        "        \\hline"
    ]
    
    # Add rows for each hyperparameter
    if embedding_dim is not None:
        latex_lines.append(f"        Embedding dim\\hfill ($d$)& {embedding_dim} \\\\")
        latex_lines.append("        \\hline")
    
    if hidden_channels is not None:
        latex_lines.append(f"        Hidden layer dim\\hfill ($H$)& {hidden_channels}  \\\\")
        latex_lines.append("        \\hline")
    
    if lr is not None:
        latex_lines.append(f"        Learning rate\\hfill ($\\gamma$) & {lr}  \\\\")
        latex_lines.append("        \\hline")
    
    if bs is not None:
        # Format batch size with comma for thousands
        bs_formatted = f"{bs:,}".replace(",", ",")
        latex_lines.append(f"        Batch size\\hfill ($m$) & {bs_formatted} \\\\")
        latex_lines.append("        \\hline")
    
    if dropout is not None:
        latex_lines.append(f"        Dropout probability\\hfill ($p$) & {dropout} \\\\")
        latex_lines.append("        \\hline")
    
    if neg_ratio is not None:
        latex_lines.append(f"        Negative sample ratio \\hfill($\\alpha$) & {neg_ratio} \\\\")
        latex_lines.append("        \\hline")
    
    if epochs is not None:
        latex_lines.append(f"        Number of epochs\\hfill ($E$) & {epochs} \\\\")
        latex_lines.append("        \\hline")
    
    # Add num_users and num_items if available (optional, not in template)
    num_users = hyperparams.get('num_users', None)
    num_items = hyperparams.get('num_items', None)
    
    # Close table
    latex_lines.append("    \\end{tabular}")
    latex_lines.append(f"    \\caption{{{caption}}}")
    latex_lines.append(f"    \\label{{{label}}}")
    latex_lines.append("\\end{table}")
    
    latex_code = "\n".join(latex_lines)
    
    return latex_code


def print_hyperparams_latex_from_file(
    checkpoint_path: str,
    **kwargs
) -> None:
    """
    Convenience function that loads checkpoint and prints LaTeX code directly.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        **kwargs: Additional hyperparameters to override/extend
    """
    latex_code = print_hyperparams_latex(checkpoint_path, **kwargs)
    print(latex_code)


if __name__ == "__main__":
    # Example usage
    import sys
    from settings import models_dir
    
    if len(sys.argv) < 2:
        print("Usage: python hyperparams.py <checkpoint_path> [learning_rate] [batch_size] [dropout_prob] [negative_sample_ratio]")
        print("\nExample:")
        print("  python hyperparams.py results/models/hetero_sage_model_approach5.pt 0.01 16384 0.5 1.0")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Parse optional arguments
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else None
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dropout = float(sys.argv[4]) if len(sys.argv) > 4 else None
    neg_ratio = float(sys.argv[5]) if len(sys.argv) > 5 else None
    
    try:
        latex_code = print_hyperparams_latex(
            checkpoint_path,
            learning_rate=lr,
            batch_size=bs,
            dropout_prob=dropout,
            negative_sample_ratio=neg_ratio
        )
        print(latex_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

