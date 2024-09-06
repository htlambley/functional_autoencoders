import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from dataclasses import field
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.util.networks import MLP


class DiracEncoder(Encoder):
    features: Sequence[int] = (128, 128, 128)
    latent_dim: int = 64
    mlp_args: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, u, x, train=False):
        u = jnp.reshape(u, (u.shape[0], -1))
        u = jnp.reshape(
            jnp.float32(jnp.argmax(u, axis=1)) / u.shape[1], (u.shape[0], 1)
        )

        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        u = MLP([*self.features, d_out], **self.mlp_args)(u)
        return u
