import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Sequence, Callable
from functional_autoencoders.decoders import Decoder
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)
from functional_autoencoders.util.networks import MLP
from dataclasses import field


class NonlinearDecoder(Decoder):
    r"""A nonlinear decoder :math:`g(z)(x) = f(z, \gamma(x))` learned using an MLP, where :math:`\gamma`
    is a positional encoding.

    The positional information :math:`\gamma(x)` of shape [batch, queries, n] is combined with the latent data
    [batch, m] by tiling the latent to shape [batch, queries, m] and concatenating to the final axis of
    :math:`\gamma(x)`, which is then fed to an MLP at the start.
    """

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    mlp_args: dict = field(default_factory=dict)
    post_activation: Callable[[ArrayLike], jax.Array] = lambda x: x
    concat_method: str = "initial"

    def _forward(self, z, x, train=False):
        x = self.positional_encoding(x)
        y = self._mlp_forward(z, x)
        y = self.post_activation(y)
        return y

    def _mlp_forward(self, z, x):
        if self.concat_method == "initial":
            return self._mlp_initial_concat(z, x)
        else:
            raise ValueError(f"Unknown method {self.method}")

    def _mlp_initial_concat(self, z, x):
        zx = self._concat(z, x)
        return MLP([*self.features, self.out_dim], **self.mlp_args)(zx)

    def _concat(self, z, x):
        n_evals = x.shape[1]
        z = jnp.repeat(jnp.expand_dims(z, 1), n_evals, axis=1)
        r = jnp.concatenate((z, x), axis=-1)
        return r
