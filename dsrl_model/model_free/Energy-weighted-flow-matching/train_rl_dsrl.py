# model_free/Energy-weighted-flow-matching/train_rl_dsrl.py

import os
import random
import time
import functools
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random
from flax import linen as nn
import optax

import gymnasium as gym
import dsrl
import dsrl.infos as dsrl_infos
import dsrl.offline_safety_gymnasium  # Registers DSRL envs

# Local imports (assumed in same package structure)
from diffusion_SDE.loss_jax import loss_fn as diffusion_loss_fn
from diffusion_SDE.schedule_jax import marginal_prob_std
from diffusion_SDE.model_jax import ScoreNetJax, update_target_jax

from dsrl_adapter import DSRLSafetyDataset
from utils import get_args
from dsrl_model.utils.logger import EpochLogger  # Assumed adapted for JAX
from dsrl_model.utils.utils import get_params_norm

EP = 1e-6

# Default configuration (Energy-Weighted Flow Matching)
default_cfg = {
    "log_freq": int(1e4),
    "save_freq": int(2e4),
    "eval_episode_freq": 10,
    "hidden_sizes": [256, 256],
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "alpha": 1.0,
    "snr_gamma": 5.0,
    "weight_decay": 0.0,
}


# Define Energy Function (MLP) in Flax
class EnergyFunctionJax(nn.Module):
    hidden_sizes: list

    @nn.compact
    def __call__(self, x):
        # MLP: hidden layers with ReLU, output single logit
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def pretrain_energy_function(params, apply_fn, opt_state, neg_obs, neg_acts, union_obs, union_acts):
    """
    JAX version of pretraining energy function.
    Args and logic are preserved: maximize energy on good trajectories, minimize on bad.
    """
    # Concatenate inputs
    neg_input = jnp.concatenate([neg_obs, neg_acts], axis=-1)
    union_input = jnp.concatenate([union_obs, union_acts], axis=-1)

    # Define loss function
    def loss_fn_energy(params):
        neg_energy = apply_fn({'params': params}, neg_input)
        union_energy = apply_fn({'params': params}, union_input)
        # Raw logits, optimize so that neg (safe) > union (mixed)
        loss = -(jnp.mean(union_energy) - jnp.mean(neg_energy))
        return loss

    grads = grad(loss_fn_energy)(params)
    # Gradient clipping
    grads = jax.tree_map(lambda g: jnp.clip(g, -default_cfg["max_grad_norm"], default_cfg["max_grad_norm"]), grads)
    updates, opt_state = energy_optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def compute_energy_weights(params, apply_fn, union_obs, union_acts):
    """
    Compute weights w = exp(alpha * E) and normalize.
    """
    inp = jnp.concatenate([union_obs, union_acts], axis=-1)
    energies = apply_fn({'params': params}, inp).squeeze()
    # Compute softmax weights (unnormalized)
    w = jnp.exp(default_cfg["alpha"] * energies)
    w_sum = jnp.sum(w) + EP
    return w / w_sum

# Main training function
def main(args):
    # Set random seeds
    rng = random.PRNGKey(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load offline safety dataset
    dset = DSRLSafetyDataset(args.dataset, args.min_cost, args.max_cost)
    obs_dim = dset.obs_dim
    act_dim = dset.act_dim

    # Environments for evaluation
    eval_env = gym.make(args.env_name)
    cost_env = gym.make(args.env_name)

    # Setup loggers
    logger = EpochLogger()
    logger.save_config = args.save_config
    logger.setup_folder(args.log_dir)

    # Network and optimizer initialization
    # Energy network
    energy_init, energy_apply = EnergyFunctionJax(hidden_sizes=args.hidden_sizes).init_with_output, None
    # Initialize parameters
    init_rng, rng = random.split(rng)
    params_energy = EnergyFunctionJax(hidden_sizes=args.hidden_sizes).init(init_rng, jnp.ones((1, obs_dim + act_dim)))['params']
    energy_optimizer = optax.adam(learning_rate=args.lr, weight_decay=args.weight_decay)
    energy_opt_state = energy_optimizer.init(params_energy)

    # Flow (policy) network
    score_net = ScoreNetJax(input_dim=obs_dim + act_dim,
                             output_dim=act_dim,
                             marginal_prob_std_fn=marginal_prob_std,
                             args=args)
    init_rng, rng = random.split(rng)
    params_flow = score_net.init(init_rng)
    flow_optimizer = optax.adam(learning_rate=args.lr, weight_decay=args.weight_decay)
    flow_opt_state = flow_optimizer.init(params_flow)
    target_params_flow = params_flow  # For target network (copy at first)

    # Policy network (same architecture as flow)
    policy_net = ScoreNetJax(input_dim=obs_dim + act_dim,
                             output_dim=act_dim,
                             marginal_prob_std_fn=marginal_prob_std,
                             args=args)
    init_rng, rng = random.split(rng)
    params_policy = policy_net.init(init_rng)
    policy_optimizer = optax.adam(learning_rate=args.lr, weight_decay=args.weight_decay)
    policy_opt_state = policy_optimizer.init(params_policy)

    # Training loop
    iters = 0
    for epoch in range(args.epochs):
        for batch in dset.get_batches(batch_size=args.batch_size, horizon=args.train_horizon):
            iters += 1
            # Prepare data batch (numpy to jax)
            obs_batch = jnp.array(batch['obs'])  # shape [batch, horizon, obs_dim]
            act_batch = jnp.array(batch['acts'])  # shape [batch, horizon, act_dim]
            # Compute energy target weights
            weights = compute_energy_weights(params_energy, energy_init, obs_batch.reshape(-1, obs_dim), act_batch.reshape(-1, act_dim))
            weights = weights.reshape(obs_batch.shape[:-1])  # [batch, horizon]

            # Update flow/policy networks using weighted flow matching loss (pseudo-code)
            def flow_loss(params_flow, params_policy, params_energy, obs, act, w):
                """
                Combined loss for flow matching and policy according to Algorithm.
                """
                # flatten trajectories
                flat_obs = obs.reshape(-1, obs.shape[-1])
                flat_act = act.reshape(-1, act.shape[-1])
                flat_w = w.reshape(-1)
                # Set condition for models
                score_net.condition = flat_obs
                policy_net.condition = flat_obs
                # Compute losses (JAX version of diffusion loss, weighted by w)
                loss_flow = diffusion_loss_fn(params_flow, flat_act, marginal_prob_std, params_energy, args.alpha, energy_apply)
                loss_flow = jnp.sum(loss_flow * flat_w)
                # Policy loss: similarly, treat policy model as diffusion model with zero energy
                loss_policy = diffusion_loss_fn(params_policy, flat_act, marginal_prob_std, None, args.alpha, energy_apply)
                loss_policy = jnp.sum(loss_policy)
                return loss_flow, loss_policy

            # Compute gradients
            grads_flow, grads_policy = grad(lambda pf, pp: flow_loss(pf, pp, params_energy, obs_batch, act_batch, weights),
                                           argnums=(0,1))(params_flow, params_policy)
            # Update parameters
            updates_flow, flow_opt_state = flow_optimizer.update(grads_flow, flow_opt_state)
            params_flow = optax.apply_updates(params_flow, updates_flow)
            updates_policy, policy_opt_state = policy_optimizer.update(grads_policy, policy_opt_state)
            params_policy = optax.apply_updates(params_policy, updates_policy)

            # Periodically update target network (soft update)
            if iters % args.target_update_freq == 0:
                target_params_flow = update_target_jax(params_flow, target_params_flow, tau=args.tau)

            # Logging
            if iters % args.log_freq == 0:
                # Compute norms
                norm_energy = get_params_norm(params_energy, grads=False)
                norm_flow = get_params_norm(params_flow, grads=False)
                norm_policy = get_params_norm(params_policy, grads=False)
                logger.log_tabular("LossFlow", loss_flow)
                logger.log_tabular("LossPolicy", loss_policy)
                logger.log_tabular("NormEnergy", norm_energy)
                logger.log_tabular("NormFlow", norm_flow)
                logger.log_tabular("NormPolicy", norm_policy)
                logger.dump_tabular()

            # Checkpointing
            if iters % args.save_freq == 0:
                os.makedirs(f"./models_rl/{args.expid}", exist_ok=True)
                # Save JAX checkpoints (using numpy save as placeholder)
                np.savez(f"./models_rl/{args.expid}/checkpoint_{iters}.npz",
                         params_energy=params_energy, params_flow=params_flow, params_policy=params_policy)

    print("Training complete.")

if __name__ == "__main__":
    args = get_args(default_cfg)
    main(args)
