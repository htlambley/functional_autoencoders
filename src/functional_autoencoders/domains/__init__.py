r"""
Domain objects specify which function space the data are assumed to come from, and encapsulate the appropriate function-space
norms and inner products needed in the FVAE and FAE losses.

- For the `fvae_sde` loss, the correct domain to use is `functional_autoencoders.domains.off_grid.SDE`.
- For the `fae` loss, we provide domain objects for:
  - functional data defined on a square domain $[0, 1]^{d}$ with zero boundary conditions, discretised on a grid (`functional_autoencoders.domains.grid.ZeroBoundaryConditions`)
  - functional data defined on a periodic domain $\mathbb{T}^{d}$, discretised on a grid (`functional_autoencoders.domains.grid.PeriodicBoundaryConditions`)
  - functional data on a square domain (with arbitrary boundary conditions), discretised on possibly non-grid meshes (`functional_autoencoders.domains.off_grid.RandomlySampledEuclidean`).

The `grid` domains allow for the assumption that the data lies in a Sobolev space of nonzero order, and the commensurate use
of Sobolev norms in the loss, whereas the `non_grid` do not permit this. 
"""

import jax
from jax.typing import ArrayLike
from typing import Tuple, Callable

NonlocalTransform = Tuple[
    Callable[[ArrayLike], jax.Array], Callable[[ArrayLike], jax.Array]
]


class Domain:
    """
    Base class representing the domain on which the data functions are defined, and the appropriate
    norms, inner products, and boundary conditions.

    Users should instantiate the appropriate derived class for their use case (see top-level `functional_autoencoders.domains`
    documentation).
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        raise NotImplementedError()

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        raise NotImplementedError()

    def nonlocal_transform(self) -> NonlocalTransform:
        raise NotImplementedError()
