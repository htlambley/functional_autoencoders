import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Sequence, Callable
from functional_autoencoders.decoders import Decoder
from functional_autoencoders.util.networks import MLP


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def _gaussian(centre, mass, std, x):
    """
    Generates a Gaussian density with mean `centre`, standard deviation `std`, mass `mass`,
    evaluated at the mesh points `x`.

    Batched over `centre`, `mass`, `std` (with the 0 axis being the batch) but using an unbatched `x`.
    """
    const = (2 * jnp.pi * std**2) ** (-0.5)
    return mass * const * jnp.exp(-((x - centre) ** 2) / (2 * std**2))


class DiracDecoder(Decoder):
    """
    Decoder that outputs a (smoothed) Dirac delta function with learned centre and mass.
    The centre and mass are computed using an MLP.
    """

    fixed_centre: bool
    features: Sequence[int] = (128, 128, 128)
    min_std: Callable[[float], float] = lambda dx: dx

    def setup(self):
        self.mlp = MLP(
            [*self.features, 2],
        )

    def _forward(self, z, x, train=False):
        if x.shape[2] != 1:
            raise NotImplementedError()

        # Implictly assumes x is the same across the batch.
        dx = x[0, 1, 0] - x[0, 0, 0]
        centre, std, mass = self.get_params(z, dx)
        return _gaussian(centre, mass, std, x[0, :, :])

    def get_params(self, z, dx):
        z = self.mlp(z)
        mass = jnp.ones_like(z[:, 0])
        if self.fixed_centre:
            c = (((1.0 / dx) // 2) + 1) / (1.0 / dx)
            centre = c * jnp.ones_like(z[:, 0])
        else:
            centre = nn.sigmoid(z[:, 0]) * (1 - 2 * dx) + dx
        std = self.min_std(dx) * jnp.ones_like(z[:, 1]) + nn.sigmoid(z[:, 1])
        return centre, std, mass
