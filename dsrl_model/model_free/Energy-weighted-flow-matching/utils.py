# model_free/Energy-weighted-flow-matching/utils.py

import numpy as np

def get_args(default_cfg):
    """
    Command-line parser producing args with default and user overrides.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Offline DSRL dataset')
    parser.add_argument('--min_cost', type=float, default=0.0)
    parser.add_argument('--max_cost', type=float, default=1.0)
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--expid', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_horizon', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--target_update_freq', type=int, default=100)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--save_config', action='store_true')
    args = parser.parse_args()
    cfg = default_cfg.copy()
    cfg.update(vars(args))
    # Convert to argparse.Namespace for attribute access
    return argparse.Namespace(**cfg)
