import flax.linen as nn
from typing import Sequence
import string
import jax.numpy as jnp

from functional_autoencoders.domains import Domain
from functional_autoencoders.util.networks import MLP, Initializer
from functional_autoencoders.domains import Domain


class FNOLayer(nn.Module):
    n_modes: Sequence[int]
    domain: Domain
    R_init: Initializer = nn.initializers.glorot_normal()
    act = nn.gelu

    @nn.compact
    def __call__(self, u):
        if u.ndim >= 25:
            raise NotImplementedError(
                "Einsum string unable to handle >= 23-dimensional domain."
            )

        transform, inverse_transform = self.domain.nonlocal_transform

        # Compute pointwise matrix multiplication
        Wu = nn.Dense(u.shape[-1], use_bias=False)(u)

        # Compute spectral convolution
        uhat = transform(u)
        kmax_shape = tuple(
            [slice(None)] + [slice(kmax) for kmax in self.n_modes] + [slice(None)]
        )
        uhat = uhat[kmax_shape]
        R_shape = [*self.n_modes, u.shape[-1], u.shape[-1]]
        R_real = self.param("R_real", self.R_init, tuple(R_shape))
        R_cplx = self.param("R_cplx", self.R_init, tuple(R_shape))
        R = R_real + 1j * R_cplx

        # Build the einsum dynamically so it works for any input dimension
        weight_shape = string.ascii_lowercase[: uhat.ndim - 2] + "xy"
        uhat_shape = "z" + string.ascii_lowercase[: uhat.ndim - 2] + "y"
        out_shape = "z" + string.ascii_lowercase[: uhat.ndim - 2] + "x"
        prod = jnp.einsum(f"{weight_shape},{uhat_shape}->{out_shape}", R, uhat)

        # Create a larger zero-padded array and transform back
        uhat_ext = jnp.zeros_like(u, dtype=jnp.complex64)
        uhat_ext = uhat_ext.at[kmax_shape].set(prod)
        Ku = inverse_transform(uhat_ext)
        return self.act(Wu + Ku)


class FNO(nn.Module):
    n_modes: Sequence[Sequence[int]]
    lifting_features: Sequence[int]
    projection_features: Sequence[int]
    domain: Domain
    act = None
    R_init: Initializer = None

    mlp_init: Initializer = None
    mlp_bias: bool = True

    @nn.compact
    def __call__(self, u, x):
        fno_kwargs = {}
        mlp_kwargs = {}

        if self.mlp_init is not None:
            mlp_kwargs["initializer"] = self.mlp_init

        if self.act is not None:
            fno_kwargs["act"] = self.act
            mlp_kwargs["act"] = self.act

        if self.R_init is not None:
            fno_kwargs["R_init"] = self.R_init

        u = MLP(self.lifting_features, **mlp_kwargs)(u)
        for layer_modes in self.n_modes:
            u = FNOLayer(layer_modes, self.domain, **fno_kwargs)(u)
        u = MLP(self.projection_features, **mlp_kwargs)(u)
        return u


class FNO1D(nn.Module):
    n_modes: Sequence[int]
    lifting_features: Sequence[int]
    projection_features: Sequence[int]
    domain: Domain
    act = None

    R_init: Initializer = None

    mlp_init: Initializer = None
    mlp_bias: bool = True

    @nn.compact
    def __call__(self, u, x):
        return FNO(
            n_modes=[(mode,) for mode in self.n_modes],
            lifting_features=self.lifting_features,
            projection_features=self.projection_features,
            domain=self.domain,
            act=self.act,
            R_init=self.R_init,
            mlp_init=self.mlp_init,
            mlp_bias=self.mlp_bias,
        )(u)
