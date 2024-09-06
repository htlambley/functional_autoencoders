import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
from dataclasses import field
from functional_autoencoders.decoders import Decoder
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)
from functional_autoencoders.util.networks import MLP


class LinearDecoder(Decoder):
    """
    Essentially the same as a "stacked" DeepONet.

    Inputs:

    `z` : [batch, basis]
      tensor of basis coefficients, e.g. [64, 10] for 10 basis coefficients

    `x` : [batch, n_evals, in_dim]
        tensor of query points
    """

    out_dim: int
    n_basis: int = 64
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    mlp_args: dict = field(default_factory=dict)

    def setup(self):
        self.net = MLP([*self.features, self.n_basis * self.out_dim], **self.mlp_args)

    def _forward(self, z, x, train=False):
        basis = self.basis(x)
        return jnp.einsum("ij,...ikjl->ikl", z, basis)

    def basis(self, x):
        x = self.positional_encoding(x)
        basis = self.net(x)
        return jnp.reshape(basis, (x.shape[0], x.shape[1], self.n_basis, self.out_dim))
