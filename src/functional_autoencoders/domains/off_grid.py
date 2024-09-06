from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


from functional_autoencoders.domains import Domain, NonlocalTransform


@partial(jax.vmap, in_axes=(0, 0))
def stochastic_integral(
    u: ArrayLike,
    v: ArrayLike,
) -> jax.Array:
    r"""
    Computes an approximation of the Itô stochastic integral

    $$ \int_{0}^{T} u_{t}^{T} \,\mathrm{d} v_{t} \approx \sum_{i} u_{t_{i}}^{T} \bigl( v_{t_{i+1}} - v_{t_{i}} \bigr)$$

    for $\mathbb{R}^{d}$-valued processes $(u_{t})_{t \in [0, T]}$ and $(v_{t})_{t \in [0, T]}$
    discretised on the same mesh. See page 45 of Särkkä and Solin (2019) for details.

    ## References
    Särkkä and Solin (2019). Applied Stochastic Differential Equations. Cambridge University Press,
      DOI: 10.1017/9781108186735.
    """

    # The sum here is both computing the outer sum over timesteps and the sum in the dot product
    # u_{t_{i}}^{T} \bigl( v_{t_{i+1}} - v_{t_{i}} \bigr)
    return jnp.sum(u[:-1, :] * jnp.diff(v, axis=0))


class RandomlySampledEuclidean(Domain):
    s: float

    def __init__(self, s: float):
        if s != 0.0:
            raise NotImplementedError()
        self.s = s
        self.name = "L^{2}"

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        return jnp.mean(jnp.sum(u**2, axis=2), axis=1)

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        return jnp.mean(jnp.sum(u * v, axis=2), axis=1)

    @property
    def nonlocal_transform(self) -> NonlocalTransform:
        raise NotImplementedError("")


class SDE(Domain):
    epsilon: float

    def __init__(self, epsilon: float, x0: float):
        self.epsilon = epsilon
        # x0 is not used directly here, but it will be accessed by the SDE loss
        self.x0 = x0
        name = "L^{2}"
        super().__init__(name)

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        dx = x[:, 1:, 0] - x[:, 0:-1, 0]
        squared_l2_norm = jnp.sum(
            jnp.sum(u[:, :-1, :] * u[:, :-1, :], axis=2) * dx, axis=1
        )
        return self.epsilon ** (-1) * squared_l2_norm

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        return self.epsilon ** (-1) * stochastic_integral(u, v)

    @property
    def nonlocal_transform(self) -> NonlocalTransform:
        raise NotImplementedError("")
