"""
Loss functions.
"""

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp


def _diag_normal_unbatched(
    key: jax.random.PRNGKey, means: ArrayLike, log_variances: ArrayLike
) -> jax.Array:
    """Generates a realisation of $N(\mu, \Sigma)$ where $\Sigma$ is a diagonal matrix of variances.

    This version is unbatched and generally the batched version `diag_normal` will be more useful.
    """
    cov_sqrt = jnp.sqrt(jnp.diag(jnp.exp(log_variances)))
    return cov_sqrt @ jax.random.normal(key, means.shape) + means


_diag_normal = jax.vmap(_diag_normal_unbatched, (0, 0, 0))


def _kl_gaussian(means, log_variances):
    """KL divergence from $N(\mu, \Sigma)$ to $N(0, I)$, when $\Sigma$ is a diagonal matrix of variances.

    The matrix $\Sigma$ is represented by an array of *log-variances* representing the diagonal of the covariance matrix.
    """
    n = means.shape[-1]
    logdets = jnp.sum(log_variances, axis=-1)
    traces = jnp.sum(jnp.exp(log_variances), axis=-1)
    return 0.5 * (-logdets - n + traces + jnp.sum(means * means, axis=-1))


def _call_autoencoder_fn(params, batch_stats, fn, u, x, name, dropout_key):
    variables = {
        "params": params[name],
        "batch_stats": (batch_stats if batch_stats else {}).get(name, {}),
    }
    result = fn(
        variables,
        u,
        x,
        train=True,
        mutable=["batch_stats"],
        rngs={"dropout": dropout_key},
    )
    return result
