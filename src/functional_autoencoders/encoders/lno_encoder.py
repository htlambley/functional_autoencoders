import flax.linen as nn
from dataclasses import field
from functional_autoencoders.domains import Domain
from functional_autoencoders.util.networks.lno import LNO
from functional_autoencoders.util.networks.pooling import MLPKernelPooling
from functional_autoencoders.encoders import Encoder, _apply_grid_encoder_operator
from functional_autoencoders.positional_encodings import (
    PositionalEncoding,
    IdentityEncoding,
)


class LNOEncoder(Encoder):
    r"""
    Encoder based on the low-rank neural operator (LNO; Kovachki et al., 2023), where each layer consists of the
    low-rank linear update

    $$v(x) = \sum_{j = 1}^{r} \langle u, \psi_{j} \rangle_{L^{2}} \phi_{j}(x),$$

    and the nonlinear update

    $$\sigma(Wu(x) + v(x)).$$

    The rank $r$ is determined by `lno_n_rank`.
    Implicitly it is assumed that the data $u$ are evaluated on a grid or a random mesh, such that the Monte Carlo approximation
    to the $L^{2}$-inner product is accurate.

    ## References
    Kovachki et al. (2023). Neural operator: learning maps between function spaces with applications to PDEs. JMLR.
    """

    latent_dim: int
    domain: Domain
    hidden_dim: int = 32
    n_layers: int = 1
    lno_n_rank: int = 12
    lno_mlp_n_layers: int = 2
    lno_mlp_n_dims: int = 32
    lno_args: dict = field(default_factory=dict)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    pooling_fn: nn.Module = MLPKernelPooling()
    pooling_concat_x: bool = True

    @nn.compact
    def __call__(self, u, x, train=False):
        lifting_features = [self.hidden_dim]
        projection_features = [self.hidden_dim]

        lno_n_ranks = [self.lno_n_rank] * self.n_layers
        lno_mlp_hidden_features = [self.lno_mlp_n_dims] * self.lno_mlp_n_layers

        operator = LNO(
            domain=self.domain,
            n_ranks=lno_n_ranks,
            lifting_features=lifting_features,
            projection_features=projection_features,
            lno_mlp_hidden_features=lno_mlp_hidden_features,
            **self.lno_args,
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
            is_grid=False,
        )
        return u
