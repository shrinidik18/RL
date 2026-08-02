# model_free/Energy-weighted-flow-matching/dsrl_adapter.py

import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import dsrl.offline_safety_gymnasium  # Registers DSRL envs

class DSRLSafetyDataset:
    """
    Adapter for DSRL offline safety datasets (NumPy-based for JAX).
    Filters trajectories based on reward and cost thresholds.
    """
    def __init__(self, env_name, min_cost, max_cost):
        self.env = gym.make(env_name)
        self.data = dsrl.load_offline_dataset(env_name)  # Pseudo-code to load
        # Filter based on cost
        if min_cost is not None or max_cost is not None:
            # Apply filtering logic to self.data
            pass

        self.obs_dim = self.data['observations'].shape[-1]
        self.act_dim = self.data['actions'].shape[-1]

    def get_batches(self, batch_size, horizon):
        """
        Generator yielding batches of shape [batch_size, horizon, dim]
        """
        obs = self.data['observations']
        acts = self.data['actions']
        N = obs.shape[0] // horizon
        # Shuffle trajectories
        idx = np.arange(N)
        np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start+batch_size]
            batch_obs = np.stack([obs[i*horizon:(i+1)*horizon] for i in batch_idx], axis=0)
            batch_acts = np.stack([acts[i*horizon:(i+1)*horizon] for i in batch_idx], axis=0)
            yield {'obs': batch_obs, 'acts': batch_acts}
