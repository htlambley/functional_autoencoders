import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from functools import partial
from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.domains import Domain
from functional_autoencoders.losses import (
    _diag_normal,
    _kl_gaussian,
    _call_autoencoder_fn,
)


def get_loss_fvae_sde_fn(
    autoencoder: Autoencoder,
    domain: Domain,
    n_monte_carlo_samples: int = 4,
    beta: float = 1,
    theta: float = 0.0,
    zero_penalty: float = 0.0,
):
    if not autoencoder.encoder.is_variational:
        raise NotImplementedError(
            "The FVAE SDE loss requires `is_variational` to be `True`"
        )

    return partial(
        _get_loss_fvae_sde,
        encode_fn=autoencoder.encoder.apply,
        decode_fn=autoencoder.decoder.apply,
        domain=domain,
        n_monte_carlo_samples=n_monte_carlo_samples,
        beta=beta,
        theta=theta,
        zero_penalty=zero_penalty,
    )


def _get_loss_fvae_sde(
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
    n_monte_carlo_samples: int,
    beta: float,
    theta: float,
    zero_penalty: float,
) -> jax.Array:
    if x_enc.shape[-1] != 1:
        raise NotImplementedError()

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
    latent_dimension = encoder_params.shape[-1] // 2

    # Generate S Monte Carlo realisations from $\mathbb{Q}_{z \mid u}^{\phi}$
    encoder_params = jnp.tile(encoder_params, (n_monte_carlo_samples, 1))
    keys = jax.random.split(key, encoder_params.shape[0])
    means = encoder_params[:, :latent_dimension]
    log_variances = encoder_params[:, latent_dimension:]
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
    decoded_grads = _decoder_grad(
        decoder_apply=decode_fn,
        out_dim=u_dec.shape[-1],
        train=True,
    )(params, batch_stats, latents, x_tile)

    u_shape = [1] * u_dec.ndim
    u_shape[0] = n_monte_carlo_samples
    u_dec = jnp.tile(u_dec, u_shape)
    norms = 0.5 * domain.squared_norm(
        decoded_grads - theta * (u_dec - decoded), x_tile
    ) - 0.5 * domain.squared_norm(theta * u_dec, x_tile)
    inner_prods = domain.inner_product(
        decoded_grads, u_dec, x_tile
    ) + domain.inner_product(theta * decoded, u_dec, x_tile)
    reconstruction_terms = norms - inner_prods
    kl_divs = beta * _kl_gaussian(means, log_variances)

    batch_stats = {
        "encoder": encoder_updates["batch_stats"],
        "decoder": decoder_updates["batch_stats"],
    }

    x0 = jnp.expand_dims(domain.x0, 0)
    x0 = jnp.repeat(x0, decoded.shape[0], axis=0)
    loss_value = (
        jnp.mean(reconstruction_terms)
        + jnp.mean(kl_divs)
        + zero_penalty * jnp.mean(jnp.sum((decoded[:, 0, :] - x0) ** 2, axis=-1))
    )

    return loss_value, batch_stats


def _decoder_grad(decoder_apply, out_dim, train):
    def inner(variables, batch_stats, z, x):
        gs = []
        for ax in range(out_dim):

            def decode(variables, batch_stats, z, x):
                batch_stats = {} if batch_stats is None else batch_stats
                return decoder_apply(
                    {
                        "params": variables["decoder"],
                        "batch_stats": batch_stats.get("decoder", {}),
                    },
                    z,
                    x,
                    train,
                )[0, 0, ax]

            g = jax.grad(decode, argnums=3)(
                variables,
                batch_stats,
                jnp.reshape(z, (1, -1)),
                jnp.reshape(x, (1, 1, 1)),
            )[0, 0, 0]

            gs.append(g)

        return jnp.array(gs)

    return jax.vmap(
        jax.vmap(inner, in_axes=(None, None, None, 0)), in_axes=(None, None, 0, 0)
    )
