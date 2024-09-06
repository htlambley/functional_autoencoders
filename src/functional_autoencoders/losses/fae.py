import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial
from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.domains import Domain


def get_loss_fae_fn(
    autoencoder: Autoencoder,
    domain: Domain,
    beta: float,
    subtract_data_norm: bool = False,
):
    if autoencoder.encoder.is_variational:
        raise NotImplementedError(
            "The FAE loss requires `is_variational` to be `False`."
        )

    return partial(
        _get_loss_fae,
        encode_fn=autoencoder.encoder.apply,
        decode_fn=autoencoder.decoder.apply,
        domain=domain,
        beta=beta,
        subtract_data_norm=subtract_data_norm,
    )


def _get_loss_fae(
    params,
    key: jax.random.PRNGKey,
    batch_stats,
    u_enc: ArrayLike,
    x_enc: ArrayLike,
    u_dec: ArrayLike,
    x_dec: ArrayLike,
    encode_fn,
    decode_fn,
    domain: Domain,
    beta: float,
    subtract_data_norm: bool,
) -> jax.Array:

    # Encode input functions u
    key, dropout_key = jax.random.split(key)
    latents, encoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=encode_fn,
        u=u_enc,
        x=x_enc,
        name="encoder",
        dropout_key=dropout_key,
    )

    # Decode latent variables
    key, dropout_key = jax.random.split(key)
    decoded, decoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=decode_fn,
        u=latents,
        x=x_dec,
        name="decoder",
        dropout_key=dropout_key,
    )

    if subtract_data_norm:
        norms = 0.5 * domain.squared_norm(decoded, x_dec)
        inner_prods = domain.inner_product(decoded, u_dec, x_dec)
        reconstruction_terms = norms - inner_prods
    else:
        reconstruction_terms = 0.5 * domain.squared_norm(decoded - u_dec, x_dec)
    regularisation_terms = beta * jnp.sum(latents**2, axis=-1)

    batch_stats = {
        "encoder": encoder_updates["batch_stats"],
        "decoder": decoder_updates["batch_stats"],
    }

    loss_value = jnp.mean(reconstruction_terms) + jnp.mean(regularisation_terms)
    return loss_value, batch_stats
