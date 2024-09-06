import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functional_autoencoders.domains import (
    Domain,
    NonlocalTransform,
)
from functional_autoencoders.util.fft import dstn, idstn


def _dstn_transform(u):
    return dstn(u, type=1, axes=range(1, u.ndim - 1), norm="forward")


def _dstn_inverse_transform(u):
    return idstn(u, type=1, axes=range(1, u.ndim - 1), norm="forward")


def _dftn_transform(u):
    return jnp.fft.fftn(u, axes=range(1, u.ndim - 1), norm="forward")


def _dftn_inverse_transform(u):
    return jnp.fft.ifftn(u, axes=range(1, u.ndim - 1), norm="forward").real


class ZeroBoundaryConditions(Domain):
    s: float

    def __init__(self, s: float):
        self.s = s
        name = f"H^{{{s}}}_{{0}}" if s != 0 else "L^{2}_{0}"
        super().__init__(name)

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        r"""Computes the squared Sobolev :math:`H^{s}` norm of a function defined on the domain :math:`[0, 1]^{d}` with zero boundary conditions.

        Given a function :math:`u \colon [0, 1]^{d} \to \Reals^{m}` of the form

        .. math
            u = \sum_{n \in \mathbb{N}^{d}} \alpha_{n} \varphi_{n}, \qquad \varphi_{n} = 2^{d/2} \prod_{i = 1}^{d} \sin(\pi n_{i} x_{i}),

        the following norm equivalent to the $H^{s}$-norm is computed:

        .. math
            \|u\|_{H^{s}([0, 1]^{d})}^{2} = \sum_{n \in \mathbb{N}^{d}} (1 + \norm{n}^{2})^{s} \norm{\alpha_{n}}^{2}.

        This will need to be noted in any write-up.

        Arguments:
        u : jnp.array of shape [batch, n_evals, out_dim]

        x: jnp.array of shape [n_evals, in_dim]

        s : float
            Sobolev exponent. The exponent $s = 0$ corresponds to the $L^{2}$ norm.

        Notes:
        If a large value of $s$ or a large number of grid points are used, this method may yield unreliable results because
        of numerical instability arising from the application of the discrete sine transform.
        This can be mitigated somewhat by passing an array `u` with double-precision floats, but for very large $s$ or grid sizes
        the results will still be incorrect even with the use of double-precision floats.
        """
        input_dimension = x.shape[-1]
        n = round(x.shape[1] ** (1 / input_dimension))
        u_shape = [u.shape[0]] + [n] * input_dimension + [u.shape[-1]]
        u = jnp.reshape(u, u_shape)

        d = u.ndim - 2
        axes = list(range(1, u.ndim - 1))
        # Scale the DST to get the coefficients in the orthonormal basis
        uhat = dstn(u, type=1, axes=axes, norm="forward") * (2 ** (d / 2))
        l2_norm_squared = jnp.sum(uhat**2, axis=-1)

        if self.s != 0.0:
            # Compute the weights $(1 + \|n\|^{2})^{s}$; when $s < 0$, nans are sometimes produced
            # as $1 + \|n\|^{2}$ is very large but in that case we replace the weights by an appropriate value.
            ax = (slice(1, sz + 1) for sz in u.shape[1:-1])
            weights = (1.0 + jnp.prod(jnp.mgrid[ax] ** 2, axis=0)) ** self.s
            weights = jnp.nan_to_num(weights, nan=jnp.inf if self.s >= 0 else 0)
            weights = jnp.expand_dims(weights, 0)
        else:
            weights = jnp.ones_like(l2_norm_squared)

        return jnp.sum(weights * l2_norm_squared, axis=range(1, weights.ndim))

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        input_dimension = x.shape[-1]
        n = round(x.shape[1] ** (1 / input_dimension))
        u_shape = [u.shape[0]] + [n] * input_dimension + [u.shape[-1]]
        u = jnp.reshape(u, u_shape)
        v = jnp.reshape(v, u_shape)

        d = u.ndim - 2
        axes = list(range(1, u.ndim - 1))
        uhat = dstn(u, type=1, axes=axes, norm="forward") * (2 ** (d / 2))
        vhat = dstn(v, type=1, axes=axes, norm="forward") * (2 ** (d / 2))

        if self.s != 0.0:
            ax = (slice(1, sz + 1) for sz in u.shape[1:-1])
            weights = (1.0 + jnp.prod(jnp.mgrid[ax] ** 2, axis=0)) ** (self.s / 2)
            weights = jnp.nan_to_num(weights, nan=jnp.inf if self.s >= 0 else 0)
            weights = jnp.expand_dims(weights, 0)
            weights = jnp.expand_dims(weights, -1)
        else:
            weights = jnp.ones_like(uhat)

        uhat = weights * uhat
        vhat = weights * vhat

        return jnp.sum(jnp.sum(uhat * vhat, axis=-1), axis=range(1, uhat.ndim - 1))

    @property
    def nonlocal_transform(self) -> NonlocalTransform:
        return (_dstn_transform, _dstn_inverse_transform)


class PeriodicBoundaryConditions(Domain):
    s: float

    def __init__(self, s: float):
        if abs(s) > 1e-6:
            # Implementing this would just be a case of swapping out `dstn` to `dftn` in `ZeroBoundaryConditions`
            raise NotImplementedError()
        self.s = s
        name = f"H^{{{s}}}_{{per}}" if s != 0 else "L^{2}_{per}"
        super().__init__(name)

    def squared_norm(self, u: ArrayLike, x: ArrayLike) -> jax.Array:
        return jnp.mean(jnp.sum(u**2, axis=2), axis=1)

    def inner_product(self, u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        return jnp.mean(jnp.sum(u * v, axis=2), axis=1)

    @property
    def nonlocal_transform(self) -> NonlocalTransform:
        return (_dftn_transform, _dftn_inverse_transform)
