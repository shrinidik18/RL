# diffusion_SDE/schedule_jax.py

import jax.numpy as jnp

def marginal_prob_std(t, schedule):
    """
    Compute mean and std of p_{0t}(x(t)|x(0)) for SDE.
    """
    if schedule == "linear":
        beta_1 = 20.0
        beta_0 = 0.1
        log_mean_coeff = -0.25 * (t ** 2) * (beta_1 - beta_0) - 0.5 * t * beta_0
        alpha_t = jnp.exp(log_mean_coeff)
        std = jnp.sqrt(1. - jnp.exp(2. * log_mean_coeff))
    elif schedule == "OT":
        alpha_t = 1 - t
        std = t
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return alpha_t, std
