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
    # TODO: This name is misleading, as it is not the number of hidden layers, but the number of hidden layers + 1 (the output layer);
    # see the definition of MLP. It is stated correctly in the paper, but the argument name is used throughout various config files
    # here so we keep it to avoid breaking things. We should change it in the future.
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

class MaxAggregation(nn.Module):
    @nn.compact
    def __call__(self, u, x):
        z = u.max(axis=1)
        return z

class IdentityMapping(nn.Module):
    @nn.compact
    def __call__(self, z):
        return z

class PoolingEncoder(Encoder):
    r"""
    Encoder used in Bunker et al. (2025), "Autoencoders in Function Space".

    In function space, the encoder is the function-to-vector operation

    $$f(u) = \rho \circ \mathrm{AGG} \circ F,$$

    where:
    - $\rho$ is a standard vector-to-vector feedforward neural network;
    - $\mathrm{AGG}$ is a function-to-vector aggregation operator; and
    - $F$ is a function-to-function operation.

    This module allows $F$ to be specified using the `F` argument, $\mathrm{AGG}$ to be specified using the `aggregation` argument,
    and, assuming `use_dense=True`, parametrises $\rho$ as

    $$ \rho(z) = W \tau(z) + b, $$

    where $\tau$ can be specified using the `tau` argument, and $W$ and $b$ are learnable parameters with dimension
    such that the output of the encoder has the correct dimension for the latent space.

    To match the setup of the paper, set:
    - `F` to `MLPPointwiseOperator()` with `mlp_dim=64` and `mlp_n_hidden_layers=3` (note that this means the MLP has 2 hidden layers plus an output layer, despite the name!),
      and `positional_encoding=RandomFourierEncoding()` with a random matrix `B` of appropriate dimension (see quickstart notebook 2);
    - `aggregation` to `MonteCarloIntegralAggregation()`; and
    - `tau` to `IdentityMapping()`. 

    This corresponds to the function-space operation

    $$ f(u) = \rho \left( \int_{\Omega} \kappa\bigl(x, u(x)\bigr) \right),$$

    where $\rho$ is a linear layer, $\kappa$ is a neural network with two hidden layers of width 64, output dimension 64, and the
    integral is discretised as the Monte Carlo sum

    $$ \int_{\Omega} \kappa\bigl(x, u(x)\bigr) \approx \frac{1}{N} \sum_{i=1}^{N} \kappa\bigl(x_i, u(x_i)\bigr).$$

    Note that $\kappa$ uses a positional encoding of $x$ with random Fourier features.

    By changing `aggregation`, it is possible to use other types of pooling, e.g., taking the $\max$ operation over the function, or
    a softmax operation, for example.
    """
    latent_dim: int
    F: nn.Module = MLPPointwiseOperator()
    aggregation: nn.Module = MonteCarloIntegralAggregation()
    tau: nn.Module = IdentityMapping()
    use_dense: bool = True

    @nn.compact
    def __call__(self, u, x, train=False):
        u = self.F(u,x) 
        z = self.aggregation(u, x)
        z = self.tau(z)
        d_out = self.latent_dim * 2 if self.is_variational else self.latent_dim
        if self.use_dense:
            z = nn.Dense(d_out)(z)
        return z
