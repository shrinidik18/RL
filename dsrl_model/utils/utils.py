# dsrl_model/utils/utils.py

import numpy as np

def get_params_norm(params, grads=False):
    """
    Compute the L2 norm of model parameters or gradients.
    """
    sq = 0.0
    for v in jax.tree_leaves(params):
        sq += jnp.sum(jnp.square(v))
    return jnp.sqrt(sq)
