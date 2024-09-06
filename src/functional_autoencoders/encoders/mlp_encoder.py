import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from dataclasses import field
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.util.networks import MLP


class MLPEncoder(Encoder):
    """A MLP-based encoder which assumes a fixed input mesh.

    Inputs:
    u : jnp.array of shape [batch, n_evals, out_dim]
        The input functions.

    x : jnp.array of shape [n_evals, in_dim]
        The input mesh, which is not actually used by the MLP encoder and is just assumed to be fixed for all
        data realisations.
    """

    features: Sequence[int] = (128, 128, 128)
    latent_dim: int = 64
    mlp_args: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, u, x, train=False):
        u = jnp.reshape(u, (u.shape[0], -1))

        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        u = MLP([*self.features, d_out], **self.mlp_args)(u)
        return u
