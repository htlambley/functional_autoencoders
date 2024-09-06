from dataclasses import dataclass
import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial


class PositionalEncoding:
    """Maps co-ordinates :math:`x` to a function :math:`\gamma(x)`.

    A positional encoding is a map :math:`\gamma(x)` mapping an `in_dim`-dimensional co-ordinate :math:`x` to an `encoding_dim`-dimensional
    positional encoding of that co-ordinate.
    Examples of positional encodings include Fourier features.

    All positional encodings should take as input:

    x : jnp.array of shape [n_evals, in_dim]

    and should return an another array of shape [n_evals, encoding_dim].
    """

    def __call__(self, x):
        raise NotImplementedError("`PositionalEncoding` must implement `__call__`.")


@dataclass
class FourierEncoding1D:
    k: int
    L: float = 1

    def __call__(self, x):
        return _fourier_features(x, self.k, self.L)


@dataclass
class RandomFourierEncoding:
    B: ArrayLike

    def __call__(self, x):
        return _random_fourier_features(x, self.B)


class IdentityEncoding(PositionalEncoding):
    def __call__(self, x):
        return x


@partial(jax.vmap, in_axes=(0, None, None))
def _fourier_features(x, k, L=1):
    r"""Computes Fourier features $[\cos(\omega x), \sin(\omega x), \dots, \cos(k\omega x), \sin(k\omega x)]$ for tensor of shape [queries, 1].

    The factor $\omega$ is given by $2\pi / L$.

    Arguments:

    x : jnp.array of shape [queries, 1]

    k : int
        number of frequencies to include in Fourier features

    L : float
        length of the one-dimensional domain

    See Seidman et al. (2023), section C.3 for implementation details; this implementation differs slightly because it omits the constant `1` in the feature array.
    """
    omega = 2 * jnp.pi / L
    a = jnp.arange(1, k + 1) * omega
    a = jnp.reshape(a, (1, -1))
    return jnp.concatenate((jnp.cos(a * x), jnp.sin(a * x)), axis=-1)


@partial(jax.vmap, in_axes=(0, None))
def _random_fourier_features(x, B):
    x = jnp.einsum("ij,qj->qi", 2 * jnp.pi * B, x)
    return jnp.concatenate((jnp.cos(x), jnp.sin(x)), axis=-1)
