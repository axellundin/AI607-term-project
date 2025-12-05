from util.graphs import FullGraph
import os 
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

curr_dir = os.path.dirname(__file__)

# Create comparison directory for outputs
comparison_dir = os.path.join(curr_dir, "comparison_results")
if not os.path.exists(comparison_dir):
    os.mkdir(comparison_dir)

def load_validation_answers(filepath):
    """Load validation answers to get label distribution"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                user_id, item_id, label = parts
                data.append({'user': user_id, 'item': item_id, 'label': int(label)})
    return data

def analyze_graph_stats(G, name):
    """Analyze comprehensive statistics of a graph"""
    stats = {
        'name': name,
        'num_users': len(G.user_to_items),
        'num_items': len(G.item_to_users),
        'num_interactions': len(G.user_item_to_interaction),
        'num_view': sum(1 for label in G.user_item_to_interaction.values() if label == 1),
        'num_save': sum(1 for label in G.user_item_to_interaction.values() if label == 2),
        'num_buy': sum(1 for label in G.user_item_to_interaction.values() if label == 3),
    }
    
    # User degree statistics
    user_degrees = [len(items) for items in G.user_to_items.values()]
    stats['avg_user_degree'] = np.mean(user_degrees)
    stats['median_user_degree'] = np.median(user_degrees)
    stats['std_user_degree'] = np.std(user_degrees)
    stats['min_user_degree'] = np.min(user_degrees)
    stats['max_user_degree'] = np.max(user_degrees)
    
    # Item degree statistics
    item_degrees = [len(users) for users in G.item_to_users.values()]
    stats['avg_item_degree'] = np.mean(item_degrees)
    stats['median_item_degree'] = np.median(item_degrees)
    stats['std_item_degree'] = np.std(item_degrees)
    stats['min_item_degree'] = np.min(item_degrees)
    stats['max_item_degree'] = np.max(item_degrees)
    
    # Sparsity
    total_possible = stats['num_users'] * stats['num_items']
    stats['sparsity'] = 1 - (stats['num_interactions'] / total_possible)
    
    return stats, user_degrees, item_degrees

def plot_degree_distributions(old_user_deg, new_user_deg, old_item_deg, new_item_deg, save_path):
    """Plot degree distributions comparing old and new datasets"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # User degree histogram
    axes[0, 0].hist(old_user_deg, bins=50, alpha=0.5, label='Old', color='blue', density=True)
    axes[0, 0].hist(new_user_deg, bins=50, alpha=0.5, label='New', color='red', density=True)
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('User Degree Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Item degree histogram
    axes[0, 1].hist(old_item_deg, bins=50, alpha=0.5, label='Old', color='blue', density=True)
    axes[0, 1].hist(new_item_deg, bins=50, alpha=0.5, label='New', color='red', density=True)
    axes[0, 1].set_xlabel('Degree')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Item Degree Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # User degree CDF
    sorted_old_user = np.sort(old_user_deg)
    sorted_new_user = np.sort(new_user_deg)
    axes[1, 0].plot(sorted_old_user, np.arange(len(sorted_old_user)) / len(sorted_old_user), 
                    label='Old', color='blue', alpha=0.7)
    axes[1, 0].plot(sorted_new_user, np.arange(len(sorted_new_user)) / len(sorted_new_user), 
                    label='New', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Degree')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 0].set_title('User Degree CDF')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Item degree CDF
    sorted_old_item = np.sort(old_item_deg)
    sorted_new_item = np.sort(new_item_deg)
    axes[1, 1].plot(sorted_old_item, np.arange(len(sorted_old_item)) / len(sorted_old_item), 
                    label='Old', color='blue', alpha=0.7)
    axes[1, 1].plot(sorted_new_item, np.arange(len(sorted_new_item)) / len(sorted_new_item), 
                    label='New', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Degree')
    axes[1, 1].set_ylabel('CDF')
    axes[1, 1].set_title('Item Degree CDF')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_label_distributions(old_stats, new_stats, save_path):
    """Plot label distribution comparison"""
    labels = ['View (1)', 'Save (2)', 'Buy (3)']
    old_counts = [old_stats['num_view'], old_stats['num_save'], old_stats['num_buy']]
    new_counts = [new_stats['num_view'], new_stats['num_save'], new_stats['num_buy']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Absolute counts
    axes[0].bar(x - width/2, old_counts, width, label='Old', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, new_counts, width, label='New', color='red', alpha=0.7)
    axes[0].set_xlabel('Interaction Type')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Interaction Type Distribution (Absolute)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (old_val, new_val) in enumerate(zip(old_counts, new_counts)):
        axes[0].text(i - width/2, old_val, f'{old_val:,}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, new_val, f'{new_val:,}', ha='center', va='bottom', fontsize=9)
    
    # Proportions
    old_total = sum(old_counts)
    new_total = sum(new_counts)
    old_props = [c / old_total for c in old_counts]
    new_props = [c / new_total for c in new_counts]
    
    axes[1].bar(x - width/2, old_props, width, label='Old', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, new_props, width, label='New', color='red', alpha=0.7)
    axes[1].set_xlabel('Interaction Type')
    axes[1].set_ylabel('Proportion')
    axes[1].set_title('Interaction Type Distribution (Proportional)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (old_val, new_val) in enumerate(zip(old_props, new_props)):
        axes[1].text(i - width/2, old_val, f'{old_val*100:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, new_val, f'{new_val*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_validation_set(old_val_path, new_val_path, name):
    """Analyze validation set label distributions including label 0"""
    old_val = load_validation_answers(old_val_path)
    new_val = load_validation_answers(new_val_path)
    
    old_label_counts = Counter([d['label'] for d in old_val])
    new_label_counts = Counter([d['label'] for d in new_val])
    
    print(f"\n{name} Label Distribution:")
    print(f"{'Dataset':<10} {'Label 0':<12} {'Label 1':<12} {'Label 2':<12} {'Label 3':<12} {'Total':<12}")
    print(f"{'Old':<10} {old_label_counts[0]:<12} {old_label_counts[1]:<12} {old_label_counts[2]:<12} {old_label_counts[3]:<12} {len(old_val):<12}")
    print(f"{'New':<10} {new_label_counts[0]:<12} {new_label_counts[1]:<12} {new_label_counts[2]:<12} {new_label_counts[3]:<12} {len(new_val):<12}")
    
    return old_label_counts, new_label_counts

def main():
    print("="*80)
    print("DATASET COMPARISON: OLD vs NEW")
    print("="*80)
    
    # Load Task 1 training data
    print("\nLoading Task 1 training data...")
    G_old_task1 = FullGraph(os.path.join(curr_dir, "../../old/old_data/task1_train.tsv"))
    G_new_task1 = FullGraph(os.path.join(curr_dir, "../../data/task1_train.tsv"))
    
    # Analyze Task 1
    print("\nAnalyzing Task 1 statistics...")
    old_stats_task1, old_user_deg_t1, old_item_deg_t1 = analyze_graph_stats(G_old_task1, "Old Task 1")
    new_stats_task1, new_user_deg_t1, new_item_deg_t1 = analyze_graph_stats(G_new_task1, "New Task 1")
    
    # Load Task 2 training data
    print("\nLoading Task 2 training data...")
    G_old_task2 = FullGraph(os.path.join(curr_dir, "../../old/old_data/task2_train.tsv"))
    G_new_task2 = FullGraph(os.path.join(curr_dir, "../../data/task2_train.tsv"))
    
    # Analyze Task 2
    print("\nAnalyzing Task 2 statistics...")
    old_stats_task2, old_user_deg_t2, old_item_deg_t2 = analyze_graph_stats(G_old_task2, "Old Task 2")
    new_stats_task2, new_user_deg_t2, new_item_deg_t2 = analyze_graph_stats(G_new_task2, "New Task 2")
    
    # Print comprehensive comparison
    print("\n" + "="*80)
    print("TASK 1 COMPARISON")
    print("="*80)
    
    comparison_keys = [
        ('num_users', 'Number of Users'),
        ('num_items', 'Number of Items'),
        ('num_interactions', 'Total Interactions'),
        ('num_view', 'View (1) Interactions'),
        ('num_save', 'Save (2) Interactions'),
        ('num_buy', 'Buy (3) Interactions'),
        ('avg_user_degree', 'Avg User Degree'),
        ('median_user_degree', 'Median User Degree'),
        ('max_user_degree', 'Max User Degree'),
        ('avg_item_degree', 'Avg Item Degree'),
        ('median_item_degree', 'Median Item Degree'),
        ('max_item_degree', 'Max Item Degree'),
        ('sparsity', 'Sparsity'),
    ]
    
    for key, label in comparison_keys:
        old_val = old_stats_task1[key]
        new_val = new_stats_task1[key]
        if key == 'sparsity':
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:.6f}   New: {new_val:.6f}   Change: {change:+.6f} ({pct_change:+.2f}%)")
        elif isinstance(old_val, float):
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:,.2f}   New: {new_val:,.2f}   Change: {change:+,.2f} ({pct_change:+.2f}%)")
        else:
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:,}   New: {new_val:,}   Change: {change:+,} ({pct_change:+.2f}%)")
    
    print("\n" + "="*80)
    print("TASK 2 COMPARISON")
    print("="*80)
    
    for key, label in comparison_keys:
        old_val = old_stats_task2[key]
        new_val = new_stats_task2[key]
        if key == 'sparsity':
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:.6f}   New: {new_val:.6f}   Change: {change:+.6f} ({pct_change:+.2f}%)")
        elif isinstance(old_val, float):
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:,.2f}   New: {new_val:,.2f}   Change: {change:+,.2f} ({pct_change:+.2f}%)")
        else:
            change = new_val - old_val
            pct_change = (change / old_val) * 100 if old_val != 0 else 0
            print(f"{label:<30} Old: {old_val:,}   New: {new_val:,}   Change: {change:+,} ({pct_change:+.2f}%)")
    
    # Analyze validation sets
    print("\n" + "="*80)
    print("VALIDATION SET COMPARISON")
    print("="*80)
    analyze_validation_set(
        os.path.join(curr_dir, "../../old/old_data/task1_val_answers.tsv"),
        os.path.join(curr_dir, "../../data/task1_val_answers.tsv"),
        "Task 1 Validation"
    )
    
    # User/Item overlap analysis
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS")
    print("="*80)
    
    old_users_t1 = set(G_old_task1.user_to_items.keys())
    new_users_t1 = set(G_new_task1.user_to_items.keys())
    old_items_t1 = set(G_old_task1.item_to_users.keys())
    new_items_t1 = set(G_new_task1.item_to_users.keys())
    
    old_users_t2 = set(G_old_task2.user_to_items.keys())
    new_users_t2 = set(G_new_task2.user_to_items.keys())
    old_items_t2 = set(G_old_task2.item_to_users.keys())
    new_items_t2 = set(G_new_task2.item_to_users.keys())
    
    print(f"\nTask 1 Users: {len(old_users_t1)} (old) vs {len(new_users_t1)} (new)")
    print(f"  - Users in both: {len(old_users_t1 & new_users_t1)}")
    print(f"  - Users only in old: {len(old_users_t1 - new_users_t1)}")
    print(f"  - Users only in new: {len(new_users_t1 - old_users_t1)}")
    
    print(f"\nTask 1 Items: {len(old_items_t1)} (old) vs {len(new_items_t1)} (new)")
    print(f"  - Items in both: {len(old_items_t1 & new_items_t1)}")
    print(f"  - Items only in old: {len(old_items_t1 - new_items_t1)}")
    print(f"  - Items only in new: {len(new_items_t1 - old_items_t1)}")
    
    print(f"\nTask 2 Users: {len(old_users_t2)} (old) vs {len(new_users_t2)} (new)")
    print(f"  - Users in both: {len(old_users_t2 & new_users_t2)}")
    print(f"  - Users only in old: {len(old_users_t2 - new_users_t2)}")
    print(f"  - Users only in new: {len(new_users_t2 - old_users_t2)}")
    
    print(f"\nTask 2 Items: {len(old_items_t2)} (old) vs {len(new_items_t2)} (new)")
    print(f"  - Items in both: {len(old_items_t2 & new_items_t2)}")
    print(f"  - Items only in old: {len(old_items_t2 - new_items_t2)}")
    print(f"  - Items only in new: {len(new_items_t2 - old_items_t2)}")
    
    # Check disjointness
    print(f"\nTask 1 and Task 2 user disjointness:")
    print(f"  Old: Task1 ∩ Task2 users = {len(old_users_t1 & old_users_t2)} (should be 0)")
    print(f"  New: Task1 ∩ Task2 users = {len(new_users_t1 & new_users_t2)} (should be 0)")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\nGenerating Task 1 degree distributions...")
    plot_degree_distributions(
        old_user_deg_t1, new_user_deg_t1,
        old_item_deg_t1, new_item_deg_t1,
        os.path.join(comparison_dir, "task1_degree_distributions.png")
    )
    
    print("Generating Task 2 degree distributions...")
    plot_degree_distributions(
        old_user_deg_t2, new_user_deg_t2,
        old_item_deg_t2, new_item_deg_t2,
        os.path.join(comparison_dir, "task2_degree_distributions.png")
    )
    
    print("Generating Task 1 label distributions...")
    plot_label_distributions(
        old_stats_task1, new_stats_task1,
        os.path.join(comparison_dir, "task1_label_distributions.png")
    )
    
    print("Generating Task 2 label distributions...")
    plot_label_distributions(
        old_stats_task2, new_stats_task2,
        os.path.join(comparison_dir, "task2_label_distributions.png")
    )
    
    # Save summary to file
    print("\nSaving summary to file...")
    summary_path = os.path.join(comparison_dir, "comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET COMPARISON SUMMARY: OLD vs NEW\n")
        f.write("="*80 + "\n\n")
        
        f.write("TASK 1 CHANGES:\n")
        f.write(f"  Total interactions: {old_stats_task1['num_interactions']:,} → {new_stats_task1['num_interactions']:,} ")
        f.write(f"({((new_stats_task1['num_interactions'] - old_stats_task1['num_interactions']) / old_stats_task1['num_interactions'] * 100):+.1f}%)\n")
        f.write(f"  Sparsity: {old_stats_task1['sparsity']:.6f} → {new_stats_task1['sparsity']:.6f}\n")
        f.write(f"  Avg user degree: {old_stats_task1['avg_user_degree']:.2f} → {new_stats_task1['avg_user_degree']:.2f}\n")
        f.write(f"  Avg item degree: {old_stats_task1['avg_item_degree']:.2f} → {new_stats_task1['avg_item_degree']:.2f}\n\n")
        
        f.write("TASK 2 CHANGES:\n")
        f.write(f"  Total interactions: {old_stats_task2['num_interactions']:,} → {new_stats_task2['num_interactions']:,} ")
        f.write(f"({((new_stats_task2['num_interactions'] - old_stats_task2['num_interactions']) / old_stats_task2['num_interactions'] * 100):+.1f}%)\n")
        f.write(f"  Sparsity: {old_stats_task2['sparsity']:.6f} → {new_stats_task2['sparsity']:.6f}\n")
        f.write(f"  Avg user degree: {old_stats_task2['avg_user_degree']:.2f} → {new_stats_task2['avg_user_degree']:.2f}\n")
        f.write(f"  Avg item degree: {old_stats_task2['avg_item_degree']:.2f} → {new_stats_task2['avg_item_degree']:.2f}\n\n")
        
        f.write("KEY INSIGHTS:\n")
        task1_increase = (new_stats_task1['num_interactions'] - old_stats_task1['num_interactions']) / old_stats_task1['num_interactions'] * 100
        task2_increase = (new_stats_task2['num_interactions'] - old_stats_task2['num_interactions']) / old_stats_task2['num_interactions'] * 100
        f.write(f"  - Task 1 has {task1_increase:.1f}% more training interactions\n")
        f.write(f"  - Task 2 has {task2_increase:.1f}% more training interactions\n")
        f.write(f"  - This provides more training data for your models\n")
        f.write(f"  - You may need to retrain models with the new dataset\n")
    
    print(f"\n{'='*80}")
    print(f"Comparison complete! Results saved to: {comparison_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
