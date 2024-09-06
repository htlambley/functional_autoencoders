import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from functional_autoencoders.decoders import Decoder
from functional_autoencoders.util.networks import MLP, CNN


class CNNDecoder(Decoder):
    c_in: int
    grid_pts_in: int
    trans_cnn_features: Sequence[int] = (32, 16, 8)
    kernel_sizes: Sequence[int] = (2, 2, 2)
    strides: Sequence[int] = (2, 2, 2)
    final_cnn_features: Sequence[int] = (16, 1)
    final_kernel_sizes: Sequence[int] = (3,)
    final_strides: Sequence[int] = (1,)
    mlp_features: Sequence[int] = (128, 128, 128)

    def _forward(self, z, x, train=False):
        u = MLP([*self.mlp_features, self.grid_pts_in**2 * self.c_in])(z)
        u = jnp.reshape(u, (-1, self.grid_pts_in, self.grid_pts_in, self.c_in))

        u = CNN(
            self.trans_cnn_features, self.kernel_sizes, self.strides, is_transpose=True
        )(u)
        u = nn.relu(u)

        u = CNN(self.final_cnn_features, self.final_kernel_sizes, self.final_strides)(u)
        u = jnp.reshape(u, (u.shape[0], -1, 1))
        return u
