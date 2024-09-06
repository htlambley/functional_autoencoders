from dataclasses import field
from functional_autoencoders.decoders import Decoder, _apply_grid_decoder_operator
from functional_autoencoders.domains import Domain
from functional_autoencoders.util.networks.fno import FNO


class FNODecoder(Decoder):
    r"""A nonlinear decoder mapping from a finite-dimensional latent variable to a function with an FNO.

    IMPORTANT: the output mesh `x` must be a regular grid on :math:`[0, 1]^{d}`, excluding the boundary.
    If `x` is incorrectly specified then internal reshapes will fail or the specified values of `x` may be silently ignored.

    First, the latent vector :math:`z \in \mathbb{R}^{\text{latent\_dim}}` is lifted to a function
    :math:`u \colon [0, 1]^{\text{in\_dim}} \to \mathbb{R}^{\text{latent\_dim} + \text{out\_dim}}` that
    takes the constant value :math:`z_{i}`, :math:`i = 1, \dots, \text{latent\_dim}` in the first `latent_dim`
    components and is the identity function in the remaining component.
    """

    out_dim: int
    domain: Domain
    hidden_dim: int = 64
    n_layers: int = 1
    n_modes_per_dim: int = 12
    fno_args: dict = field(default_factory=dict)

    def _forward(self, z, x, train=False):
        n_modes = [[self.n_modes_per_dim] * x.shape[-1]] * self.n_layers
        lifting_features = [self.hidden_dim]
        projection_features = [self.hidden_dim]

        operator = FNO(
            n_modes,
            lifting_features,
            [*projection_features, self.out_dim],
            self.domain,
            **self.fno_args,
        )
        u = _apply_grid_decoder_operator(z, x, operator)
        return u
