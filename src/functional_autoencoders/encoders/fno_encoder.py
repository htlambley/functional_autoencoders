import flax.linen as nn
from dataclasses import field
from functional_autoencoders.domains import Domain
from functional_autoencoders.util.networks.fno import FNO
from functional_autoencoders.util.networks.pooling import MLPKernelPooling
from functional_autoencoders.encoders import Encoder, _apply_grid_encoder_operator
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)


class FNOEncoder(Encoder):
    """
    An FNO-based encoder.

    IMPORTANT: inputs must be provided on a regular grid for use with the FNO.
    See notes below.

    Inputs:
    u : jnp.array of shape [batch, n_evals, out_dim]
        Will be reshaped internally to shape [batch, ax_1, ..., ax_d, out_dim],
        where `d` is `in_dim` from the shape of `x`, and `ax_1 = ax_2 = ... = ax_d = n_evals ** (1/d)`.
        If `u` is not evaluated on a regular grid, this will either fail to reshape or incorrectly treat the
        data as if it were on a grid.

    x : jnp.array of shape [batch, n_evals, in_dim]
        The mesh on which the inputs are evaluated. Must be a regular grid in the domain $[0, 1]^{d}$,
        excluding the boundary.
    """

    latent_dim: int
    domain: Domain
    fno_lifting_dim: int = 32
    fno_projection_dim: int = 4
    n_modes_per_dim: int = 12
    kernel_hidden_dim: int = 32
    kernel_n_layers: int = 2
    post_kernel_hidden_dim: int = 16
    post_kernel_n_layers: int = 3
    n_layers: int = 1
    fno_args: dict = field(default_factory=dict)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    pooling_fn: nn.Module = MLPKernelPooling()
    pooling_concat_x: bool = True

    @nn.compact
    def __call__(self, u, x, train=False):
        n_modes = [[self.n_modes_per_dim] * x.shape[-1]] * self.n_layers
        lifting_features = [self.fno_lifting_dim]
        projection_features = [self.fno_projection_dim]

        operator = FNO(
            n_modes,
            lifting_features,
            projection_features,
            self.domain,
            **self.fno_args,
        )

        u = _apply_grid_encoder_operator(
            u=u,
            x=x,
            x_pos=self.positional_encoding(x),
            operator=operator,
            pooling_fn=self.pooling_fn,
            latent_dim=self.latent_dim,
            is_variational=self.is_variational,
            is_concat=self.pooling_concat_x,
            is_grid=True,
        )
        return u
