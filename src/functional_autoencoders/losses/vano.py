import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial
from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.losses import (
    _diag_normal,
    _kl_gaussian,
    _call_autoencoder_fn,
)


def get_loss_vano_fn(
    autoencoder: Autoencoder,
    n_monte_carlo_samples: int = 4,
    beta: float = 1,
    normalised_inner_prod: bool = True,
    rescale_by_norm: bool = True,
):
    r"""
    Computes the VANO loss of Seidman et al. (2023), which corresponds to the FVAE loss with white noise $\eta \sim N(0, I)$
    in the VAE decoder model.

    Notes:
    - To follow the loss described by Seidman et al. (2023), set `normalised_inner_prod` to `False`, which corresponds to not normalising by :math:`1/N_{\text{points}}` in :math:`(\dagger)`.
    This choice can lead to instability across resolutions as the inner product is no longer correctly normalised but it may be useful for comparison.
    """
    if not autoencoder.encoder.is_variational:
        raise NotImplementedError(
            "The VANO loss requires `is_variational` to be `True`"
        )

    return partial(
        _get_loss_vano,
        encode_fn=autoencoder.encoder.apply,
        decode_fn=autoencoder.decoder.apply,
        n_monte_carlo_samples=n_monte_carlo_samples,
        beta=beta,
        normalised_inner_prod=normalised_inner_prod,
        rescale_by_norm=rescale_by_norm,
    )


def _get_loss_vano(
    params,
    key: jax.random.PRNGKey,
    batch_stats,
    u_enc: ArrayLike,
    x_enc: ArrayLike,
    u_dec: ArrayLike,
    x_dec: ArrayLike,
    encode_fn,
    decode_fn,
    n_monte_carlo_samples: int,
    beta: float,
    normalised_inner_prod: bool,
    rescale_by_norm: bool,
) -> jax.Array:

    scales = jnp.ones((u_dec.shape[0],))
    if rescale_by_norm:
        scales = (
            jnp.mean(jnp.sum(u_dec**2, axis=-1), axis=range(1, u_dec.ndim - 1))
        ) ** (-0.5)
    scales = jnp.tile(scales, (n_monte_carlo_samples,))

    # Encode input functions u
    key, dropout_key = jax.random.split(key)
    encoder_params, encoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=encode_fn,
        u=u_enc,
        x=x_enc,
        name="encoder",
        dropout_key=dropout_key,
    )
    latent_dim = encoder_params.shape[-1] // 2

    # Generate S Monte Carlo realisations from $\mathbb{Q}_{z \mid u}^{\phi}$
    encoder_params = jnp.tile(encoder_params, (n_monte_carlo_samples, 1))
    keys = jax.random.split(key, encoder_params.shape[0])
    means = encoder_params[:, :latent_dim]
    log_variances = encoder_params[:, latent_dim:]
    latents = _diag_normal(keys, means, log_variances)

    # Decode the S Monte Carlo realisations
    tiling_shape = [n_monte_carlo_samples] + [1] * (x_dec.ndim - 1)
    x_tile = jnp.tile(x_dec, tiling_shape)

    key, dropout_key = jax.random.split(key)
    decoded, decoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=decode_fn,
        u=latents,
        x=x_tile,
        name="decoder",
        dropout_key=dropout_key,
    )

    # Estimate half squared L^{2} norm of each D_{\theta}(z) of shape [batch * S, points, out_dim]
    norms = 0.5 * jnp.mean(jnp.sum(decoded**2, axis=2), axis=1)

    # Tile the true data $u$ to have the same shape
    u_dec = jnp.tile(u_dec, (n_monte_carlo_samples, 1, 1))

    # Estimate inner product term, optionally normalising (see documentation of this function for rationale)
    inner_prods = jnp.sum(decoded * u_dec, axis=(1, 2))
    if normalised_inner_prod:
        inner_prods /= u_dec.shape[1]

    # Explicit formula for KL divergence for multivariate Gaussians with identity covariance
    # and different means
    reconstruction_terms = scales * (norms - inner_prods)
    kl_divs = beta * _kl_gaussian(means, log_variances)

    batch_stats = {
        "encoder": encoder_updates["batch_stats"],
        "decoder": decoder_updates["batch_stats"],
    }

    loss_value = jnp.mean(reconstruction_terms) + jnp.mean(kl_divs)
    return loss_value, batch_stats
