#!/usr/bin/env python3
"""
Visualize cost vs returns for DSRL dataset trajectories across different environments.
Creates scatter plots showing the distribution of trajectories in the cost-return space.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import gymnasium as gym
import dsrl
import dsrl.offline_safety_gymnasium

from dsrl_dataset import (
    get_dataset_in_d4rl_format,
    get_neg_and_union_data_2,
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def compute_trajectory_statistics(data_dict):
    """
    Compute returns and costs for each trajectory.
    
    Args:
        data_dict: Dictionary with keys ['observations', 'actions', 'rewards', 'costs', ...]
                   Each is a numpy array of shape [num_trajectories, ep_len, dim]
    
    Returns:
        returns: array of shape [num_trajectories]
        costs: array of shape [num_trajectories]
    """
    rewards = data_dict['rewards']  # [num_traj, ep_len, 1] or [num_traj, ep_len]
    costs = data_dict['costs']      # [num_traj, ep_len, 1] or [num_traj, ep_len]
    
    # Debug prints
    print(f"  Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
    print(f"  Costs shape: {costs.shape}, dtype: {costs.dtype}")
    
    # Handle different shapes
    if rewards.ndim == 3:
        rewards = rewards.squeeze(-1)
    if costs.ndim == 3:
        costs = costs.squeeze(-1)
    
    # Ensure we're working with float arrays
    rewards = rewards.astype(np.float32)
    costs = costs.astype(np.float32)
    
    # Sum over episode length
    trajectory_returns = rewards.sum(axis=1)
    trajectory_costs = costs.sum(axis=1)
    
    print(f"  Computed returns shape: {trajectory_returns.shape}")
    print(f"  Computed costs shape: {trajectory_costs.shape}")
    
    return trajectory_returns, trajectory_costs


def visualize_environment(task_name, dataset_config, save_dir='./plots'):
    """
    Create scatter plot of cost vs returns for a specific environment.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {task_name}")
    print(f"{'='*60}")
    
    # Create environment
    env = gym.make(task_name)
    env.reset(seed=42)
    
    # Get raw dataset
    raw_data = env.get_dataset()
    
    # Compute trajectory lengths
    dones_idx = np.where((raw_data["terminals"] == 1) | (raw_data["timeouts"] == 1))[0]
    traj_lengths = []
    start = 0
    for end_idx in dones_idx:
        traj_lengths.append(end_idx - start + 1)
        start = end_idx + 1
    
    max_traj_len = max(traj_lengths)
    mean_traj_len = np.mean(traj_lengths)
    print(f"Trajectory stats: mean={mean_traj_len:.1f}, max={max_traj_len}, min={min(traj_lengths)}")
    
    ep_len = max_traj_len
    
    # Convert to d4rl format
    d4rl_data = get_dataset_in_d4rl_format(
        env=env,
        config=dataset_config,
        task=task_name,
        ep_len=ep_len,
        num_folds=1
    )
    
    print(f"D4RL data shape: {d4rl_data['observations'].shape}")
    
    # Get negative and union splits
    try:
        neg_data, union_data = get_neg_and_union_data_2(d4rl_data, dataset_config)
    except Exception as e:
        print(f"Error in get_neg_and_union_data_2: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"Negative set: {neg_data['observations'].shape}")
    print(f"Union set: {union_data['observations'].shape}")
    
    # Verify data types
    print(f"Negative data types:")
    for key in ['observations', 'actions', 'rewards', 'costs']:
        if key in neg_data:
            print(f"  {key}: shape={neg_data[key].shape}, dtype={neg_data[key].dtype}")
    
    print(f"Union data types:")
    for key in ['observations', 'actions', 'rewards', 'costs']:
        if key in union_data:
            print(f"  {key}: shape={union_data[key].shape}, dtype={union_data[key].dtype}")
    
    # Compute statistics
    neg_returns, neg_costs = compute_trajectory_statistics(neg_data)
    union_returns, union_costs = compute_trajectory_statistics(union_data)
    
    print(f"\nNegative trajectories:")
    print(f"  Returns: mean={neg_returns.mean():.2f}, std={neg_returns.std():.2f}, "
          f"min={neg_returns.min():.2f}, max={neg_returns.max():.2f}")
    print(f"  Costs: mean={neg_costs.mean():.2f}, std={neg_costs.std():.2f}, "
          f"min={neg_costs.min():.2f}, max={neg_costs.max():.2f}")
    
    print(f"\nUnion trajectories:")
    print(f"  Returns: mean={union_returns.mean():.2f}, std={union_returns.std():.2f}, "
          f"min={union_returns.min():.2f}, max={union_returns.max():.2f}")
    print(f"  Costs: mean={union_costs.mean():.2f}, std={union_costs.std():.2f}, "
          f"min={union_costs.min():.2f}, max={union_costs.max():.2f}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Combine all trajectories (negative + union) and plot as one set
    all_returns = np.concatenate([neg_returns, union_returns])
    all_costs = np.concatenate([neg_costs, union_costs])
    
    # Plot all trajectories as blue circles (SWAPPED: cost on x-axis, return on y-axis)
    ax.scatter(all_costs, all_returns, 
               alpha=0.6, s=60, c='steelblue', 
               label=f'All Trajectories ({len(all_returns)})', 
               edgecolors='navy', linewidth=0.5)
    
    # Add reference lines (SWAPPED)
    ax.axhline(y=np.median(all_returns), color='gray', linestyle='--', 
               alpha=0.3, linewidth=1, label='Median return')
    ax.axvline(x=np.median(all_costs), color='gray', linestyle='--', 
               alpha=0.3, linewidth=1, label='Median cost')
    
    # Labels and title (SWAPPED)
    ax.set_xlabel('Episode Cost (Cumulative)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Return (Cumulative Reward)', fontsize=12, fontweight='bold')
    
    # Clean task name for title
    clean_name = task_name.replace('Offline', '').replace('Gymnasium-v1', '').replace('Velocity', ' Velocity')
    ax.set_title(f'Cost vs Return Distribution: {clean_name}', fontsize=14, fontweight='bold')
    
    # Add text box with statistics
    stats_text = (
        f"Total Trajectories: {len(all_returns)}\n"
        f"(Negative: {len(neg_returns)}, Union: {len(union_returns)})\n"
        f"Return: μ={all_returns.mean():.1f}, σ={all_returns.std():.1f}\n"
        f"Cost: μ={all_costs.mean():.1f}, σ={all_costs.std():.1f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9, family='monospace')
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    safe_filename = task_name.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(save_dir, f'{safe_filename}_cost_vs_return.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    plt.close()
    
    return {
        'task': task_name,
        'neg_returns': neg_returns,
        'neg_costs': neg_costs,
        'union_returns': union_returns,
        'union_costs': union_costs,
    }


def create_combined_plot(all_results, save_dir='./plots'):
    """
    Create a combined multi-panel plot showing all environments.
    """
    n_envs = len(all_results)
    n_cols = 2
    n_rows = (n_envs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten() if n_envs > 1 else [axes]
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        
        # Extract data
        task = result['task']
        neg_returns = result['neg_returns']
        neg_costs = result['neg_costs']
        union_returns = result['union_returns']
        union_costs = result['union_costs']
        
        # Combine negative + union
        all_returns = np.concatenate([neg_returns, union_returns])
        all_costs = np.concatenate([neg_costs, union_costs])
        
        # Plot all as uniform blue circles (SWAPPED: cost on x, return on y)
        ax.scatter(all_costs, all_returns, 
                   alpha=0.6, s=40, c='steelblue', 
                   label=f'Trajectories ({len(all_returns)})', 
                   edgecolors='navy', linewidth=0.3)
        
        # Reference lines (SWAPPED)
        ax.axhline(y=np.median(all_returns), color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=np.median(all_costs), color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Labels (SWAPPED)
        clean_name = task.replace('Offline', '').replace('Gymnasium-v1', '').replace('Velocity', '')
        ax.set_title(clean_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Cost', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_envs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cost vs Return Across DSRL Environments', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, 'combined_all_environments.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined plot saved to: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize DSRL dataset cost vs returns')
    parser.add_argument('--envs', '--environments', nargs='+', 
                        default=None,
                        help='Specific environments to analyze (e.g., --envs OfflineSwimmerVelocityGymnasium-v1)')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all available environments')
    parser.add_argument('--save_dir', type=str, default='./dataset_analysis_plots',
                        help='Directory to save plots')
    parser.add_argument('--num_negative', type=int, default=50,
                        help='Number of negative trajectories')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Density for dataset preprocessing')
    parser.add_argument('--with_inpainting', action='store_true', default=False,
                        help='Enable inpainting (default: disabled)')
    args = parser.parse_args()
    
    # All available DSRL environments
    all_environments = [
        "OfflinePointGoal1Gymnasium-v0",
        "OfflinePointButton1Gymnasium-v0",
        "OfflineSwimmerVelocityGymnasium-v1",
        "OfflineAntVelocityGymnasium-v1",
        "OfflineCarGoal1Gymnasium-v0",
        "OfflineCarButton1Gymnasium-v0",
        "OfflineHalfCheetahVelocityGymnasium-v1",
        "OfflineWalker2dVelocityGymnasium-v1",
        "OfflineHopperVelocityGymnasium-v1",
    ]
    
    # Select environments based on arguments
    if args.envs:
        environments = args.envs
        print(f"Analyzing specified environments: {environments}")
    elif args.all:
        environments = all_environments
        print(f"Analyzing all {len(environments)} environments")
    else:
        # Default subset for quick analysis
        environments = [
            "OfflinePointGoal1Gymnasium-v0",
            "OfflineSwimmerVelocityGymnasium-v1",
            "OfflineAntVelocityGymnasium-v1",
        ]
        print(f"Analyzing default environments: {environments}")
        print("Use --all to analyze all environments, or --envs to specify particular ones")
    
    
    # Dataset configuration (matching IPLtwin.py defaults)
    inpaint = ((0.0, 1.0, 0.0, 0.5),) if args.with_inpainting else None
    
    dataset_config = {
        "density": args.density,
        "inpaint_ranges": inpaint,
        "num_negative_trajectories": args.num_negative,
        "num_union_trajectories": -1,  # Use all remaining
        "non_pref_noise": 0.0,
    }
    
    save_dir = args.save_dir
    print(f"\nConfiguration:")
    print(f"  Save directory: {save_dir}")
    print(f"  Num negative trajectories: {args.num_negative}")
    print(f"  Density: {args.density}")
    print(f"  Inpainting: {'ENABLED' if args.with_inpainting else 'DISABLED'}\n")
    
    all_results = []
    
    for task in environments:
        try:
            result = visualize_environment(task, dataset_config, save_dir)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {task}: {e}")
            continue
    
    # Create combined plot
    if all_results:
        create_combined_plot(all_results, save_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Processed {len(all_results)} environments.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()