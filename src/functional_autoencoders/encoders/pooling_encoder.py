import flax.linen as nn
import jax.numpy as jnp
from functional_autoencoders.util.networks.pooling import DeepSetPooling
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)


class PoolingEncoder(Encoder):
    latent_dim: int
    pooling_fn: nn.Module = DeepSetPooling()
    positional_encoding: PositionalEncoding = IdentityEncoding()

    @nn.compact
    def __call__(self, u, x, train=False):
        x_pos = self.positional_encoding(x)

        u = jnp.concatenate([x_pos, u], axis=-1)
        z = self.pooling_fn(u, x_pos)

        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        z = nn.Dense(d_out)(z)
        return z
