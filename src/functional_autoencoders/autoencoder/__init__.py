import flax.linen as nn
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.decoders import Decoder


class Autoencoder(nn.Module):
    r"""A flexible autoencoder designed for operator encoders and decoders.

    All encoders should take inputs:
        `u` : jnp.array of shape [batch, n_evals, out_dim]
        Represents `n_evals` evaluations of functions :math:`u \colon [0, 1]^{\text{in\_dim}} \to \mathbb{R}^{\text{out\_dim}}`.
        The evaluations do NOT include the boundary of :math:`[0,1]^{\text{in\_dim}}`

        `x` : jnp.array of shape [n_evals, in_dim]
        Represents the mesh points upon which :math:`u` is evaluated.

    Encoders should return an array of shape [batch, 2 * latent_dim], with the first latent_dim components representing the encoder
    mean and the second latent_dim components representing the log-variances on the diagonal of the encoder covariance.

    All decoders should take inputs:
        `z` : jnp.array of shape [batch, latent_dim]
        Represents the latent variables

        `x` : jnp.array of shape [n_evals, in_dim]
        Represents the mesh grids to evaluate the output function.

    Decoders should return an array of the same shape as the input `u` to the encoder.

    Notes:
    - It is assumed that the input mesh and the output mesh are the same, and that the mesh is the same for each example in
      the batch, so `x` does *not* have a batch dimension.

    - Calling the `Autoencoder` directly will map (u, x) to the latents z using the encoder, then (without adding the encoder noise)
      map straight back using the decoder (without adding any decoder noise).
    """

    encoder: Encoder
    decoder: Decoder

    @nn.compact
    def __call__(self, u, x_enc, x_dec, train=False):
        z = self.encoder(u, x_enc, train)

        if self.encoder.is_variational:
            latent_dim = self.get_latent_dim()
            mean, _ = (
                z[:, :latent_dim],
                z[:, latent_dim:],
            )
            return self.decoder(mean, x_dec, train)
        else:
            return self.decoder(z, x_dec, train)

    def encode(self, state, u, x, train=False):
        return self.encoder.apply(
            {
                "params": state.params["encoder"],
                "batch_stats": state.batch_stats["encoder"],
            },
            u,
            x,
            train,
        )

    def decode(self, state, z, x, train=False):
        return self.decoder.apply(
            {
                "params": state.params["decoder"],
                "batch_stats": state.batch_stats["decoder"],
            },
            z,
            x,
            train,
        )

    def get_latent_dim(self):
        return self.encoder.get_latent_dim()
