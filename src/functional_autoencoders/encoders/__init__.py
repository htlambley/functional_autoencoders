import jax.numpy as jnp
import flax.linen as nn


class Encoder(nn.Module):
    """
    For general comments, see `Autoencoder` documentation.
    """

    is_variational: bool

    def __call__(self, u, x):
        raise NotImplementedError()

    def get_latent_dim(self):
        return self.latent_dim


def _apply_grid_encoder_operator(
    u, x, x_pos, operator, latent_dim, is_variational, pooling_fn, is_concat, is_grid
):
    if is_grid:
        input_dimension = x.shape[-1]
        n = round(x.shape[1] ** (1 / input_dimension))
        x_shape = [u.shape[0]] + [n] * input_dimension + [x.shape[-1]]
        u_shape = [u.shape[0]] + [n] * input_dimension + [u.shape[-1]]
        x = jnp.reshape(x, x_shape)
        u = jnp.reshape(u, u_shape)

    u = operator(u, x)

    if is_concat:
        u = jnp.concatenate([u, x], axis=-1)

    u = pooling_fn(u, x_pos)
    d_out = latent_dim * 2 if is_variational else latent_dim
    u = nn.Dense(d_out, use_bias=False)(u)
    return u
