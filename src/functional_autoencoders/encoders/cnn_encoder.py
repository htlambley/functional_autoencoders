import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.util.networks import CNN, MLP


class CNNEncoder(Encoder):
    """A CNN-based encoder which assumes a fixed input mesh.

    Inputs:
    u : jnp.array of shape [batch, n_evals, out_dim]
        The input functions.

    x : jnp.array of shape [n_evals, in_dim]
        The input mesh, which is not actually used by the MLP encoder and is just assumed to be fixed for all
        data realisations.
    """

    cnn_features: Sequence[int] = (8, 16, 32)
    mlp_features: Sequence[int] = (128, 128, 128)
    kernel_sizes: Sequence[int] = (2, 2, 2)
    strides: Sequence[int] = (2, 2, 2)
    latent_dim: int = 64

    @nn.compact
    def __call__(self, u, x, train=False):
        n = int(u.shape[1] ** 0.5)
        u = jnp.reshape(u, (-1, n, n, 1))

        u = CNN(self.cnn_features, self.kernel_sizes, self.strides)(u)
        u = jnp.reshape(u, (u.shape[0], -1))

        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        u = MLP([*self.mlp_features, d_out])(u)
        return u
