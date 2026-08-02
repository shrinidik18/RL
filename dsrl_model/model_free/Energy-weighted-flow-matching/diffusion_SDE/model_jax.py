# diffusion_SDE/model_jax.py

import jax.numpy as jnp
from flax import linen as nn
import jax

class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.0

    def setup(self):
        # Random Fourier features (non-trainable)
        self.W = self.param('W', jax.random.normal, (self.embed_dim // 2,)) * self.scale

    def __call__(self, t):
        """
        t: shape [..., 1]
        Returns concatenated sin, cos features of shape [..., embed_dim]
        """
        t_proj = t[..., None] * self.W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)


class ResidualBlock(nn.Module):
    input_dim: int
    output_dim: int
    t_dim: int = 128

    def setup(self):
        self.time_mlp = nn.Sequential([nn.silu, nn.Dense(self.output_dim)])
        self.dense1 = nn.Sequential([nn.Dense(self.output_dim), nn.silu])
        self.dense2 = nn.Sequential([nn.Dense(self.output_dim), nn.silu])
        if self.input_dim != self.output_dim:
            self.modify_x = nn.Dense(self.output_dim)
        else:
            self.modify_x = lambda x: x

    def __call__(self, x, t_embed):
        h1 = self.dense1(x) + self.time_mlp(t_embed)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)


class ScoreBaseJax(nn.Module):
    input_dim: int
    output_dim: int
    marginal_prob_std: callable  # function
    embed_dim: int = 32
    args: object = None

    def setup(self):
        self.embed = GaussianFourierProjection(self.embed_dim)
        # Downsample/upsample blocks
        self.pre_sort_condition = nn.Sequential([nn.Dense(32), nn.silu])  # Equivalent to (input_dim-output_dim) ->32, SiLU
        self.sort_t = nn.Sequential([nn.Dense(128), nn.silu, nn.Dense(128)])  # 64->128->128
        self.down_block1 = ResidualBlock(self.output_dim, 512)
        self.down_block2 = ResidualBlock(512, 256)
        self.down_block3 = ResidualBlock(256, 128)
        self.middle = ResidualBlock(128, 128)
        self.up_block3 = ResidualBlock(256, 256)
        self.up_block2 = ResidualBlock(512, 512)
        self.last = nn.Dense(self.output_dim)

        self.condition = None

    def __call__(self, x, t, condition=None):
        # x: [batch, dim], t: [batch], condition: [batch, cond_dim]
        # Embed time
        t_embed = self.embed(t)  # shape [batch, embed_dim]
        if condition is not None:
            cond_embed = self.pre_sort_condition(condition)
            t_embed = jnp.concatenate([cond_embed, t_embed], axis=-1)
        # Downsample blocks
        d1 = self.down_block1(x, t_embed)
        d2 = self.down_block2(d1, t_embed)
        d3 = self.down_block3(d2, t_embed)
        # Middle block
        mid = self.middle(d3, t_embed)
        # Upsample blocks
        u3 = self.up_block3(jnp.concatenate([d3, mid], axis=-1), t_embed)
        u2 = self.up_block2(jnp.concatenate([d2, u3], axis=-1), t_embed)
        u1 = self.last(jnp.concatenate([d1, u2], axis=-1))
        # Normalize by noise std
        std = self.marginal_prob_std(t)[1]
        std = std[..., None]
        return u1 / std


class ScoreNetJax(ScoreBaseJax):
    """
    ScoreNet architecture (inherited from ScoreBaseJax).
    Acts as our flow and policy model.
    """
    pass  # All behavior from ScoreBaseJax; separate class for semantic clarity


def update_target_jax(params, target_params, tau):
    """
    Polyak averaging for target network parameters.
    """
    return jax.tree_multimap(lambda p, tp: tp + tau * (p - tp), params, target_params)
