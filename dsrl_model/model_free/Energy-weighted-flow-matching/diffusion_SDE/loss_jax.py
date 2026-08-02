# diffusion_SDE/loss_jax.py

import jax
import jax.numpy as jnp

def loss_fn(params_model, x, marginal_prob_std_fn, energy, alpha, energy_apply_fn, eps=1e-3):
    """
    JAX version of diffusion SDE loss.
    If energy is None, compute standard score matching loss.
    If energy provided, compute weighted loss.
    """
    rng = jax.random.PRNGKey(0)
    if energy is None:
        # x shape: [batch, dim]
        batch = x.shape[0]
        rng, step_rng = jax.random.split(rng)
        t = jax.random.uniform(step_rng, shape=(batch,), minval=eps, maxval=1.0)
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, shape=x.shape)
        alpha_t, std = marginal_prob_std_fn(t)
        alpha_t = alpha_t[:, None]
        std = std[:, None]
        perturbed = x * alpha_t + z * std
        score = energy_apply_fn({'params': params_model}, perturbed, t) if energy_apply_fn else None
        # Since no energy guidance, treat energy as None in call
        score = energy_apply_fn({'params': params_model}, perturbed, t)
        loss = jnp.mean(jnp.sum((score * std + z) ** 2, axis=1))
    else:
        # x shape: [horizon, batch, dim] or [batch, horizon, dim] flattened to [batch*hor, dim]
        shape = x.shape
        t_shape = shape[:-1]  # e.g., [horizon, batch]
        rng, step_rng = jax.random.split(rng)
        t = jax.random.uniform(step_rng, shape=t_shape, minval=eps, maxval=1.0)
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, shape=x.shape)
        alpha_t, std = marginal_prob_std_fn(t)
        alpha_t = alpha_t[..., None]
        std = std[..., None]
        perturbed = x * alpha_t + z * std
        score = energy_apply_fn({'params': params_model}, perturbed, t)
        # energy: array shaped [horizon, batch] or [batch*hor]
        guidance = jax.nn.softmax(energy * alpha, axis=0).squeeze()
        individual_loss = jnp.sum((score * std + z) ** 2, axis=1)  # sum over dim
        loss = jnp.dot(individual_loss, guidance)
    return loss
