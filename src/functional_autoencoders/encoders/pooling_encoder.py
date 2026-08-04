from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from functional_autoencoders.util.networks import MLP
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)

class MLPPointwiseOperator(nn.Module):
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    positional_encoding: PositionalEncoding = IdentityEncoding()

    @nn.compact
    def __call__(self, u, x):
        x_pos = self.positional_encoding(x)
        u = jnp.concatenate([x_pos, u], axis=-1)
        u = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)
        return u


class MonteCarloIntegralAggregation(nn.Module):
    @nn.compact
    def __call__(self, u, x):
        z = u.mean(axis=1)
        return z

class IdentityMapping(nn.Module):
    @nn.compact
    def __call__(self, z):
        return z

class PoolingEncoder(Encoder):
    r"""
    Encoder used in Bunker et al. (2025), "Autoencoders in Function Space".

    In function space, the encoder is the function-to-vector operation

    $$f(u) = p \circ \rho \circ \mathrm{AGG} \circ F,$$

    where:
    - $p$ is a learnable linear projection to the latent dimension;
    - $\rho$ is a standard vector-to-vector feedforward neural network;
    - $\mathrm{AGG}$ is a function-to-vector aggregation operator---in the original architecture, $\mathrm{AGG}(u) = \int_{\Omega} u(x) dx$;
    - $F$ is a function-to-function operation---in the original architecture, a standard vector-to-vector neural network applied pointwise;

    TODO: customise rho, kappa architecture
    TODO: customise discretisation of integral
    """
    latent_dim: int
    F: nn.Module = MLPPointwiseOperator()
    aggregation: nn.Module = MonteCarloIntegralAggregation()
    rho: nn.Module = IdentityMapping()

    @nn.compact
    def __call__(self, u, x, train=False):
        u = self.F(u,x) 
        z = self.aggregation(u, x)
        z = self.rho(z)
        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        z = nn.Dense(d_out)(z)
        return z
