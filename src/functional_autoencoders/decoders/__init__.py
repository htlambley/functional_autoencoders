import flax.linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    """
    For general comments, see `Autoencoder` documentation.
    """

    @nn.compact
    def __call__(self, z, x, train=False):
        u = self._forward(z, x, train)
        return u

    def _forward(self, z, x):
        raise NotImplementedError()


def _apply_grid_decoder_operator(z, x, operator):
    """
    Helper function to reshape appropriately to a grid in order to apply grid-based operators like
    the Fourier neural operator, used in `FNODecoder`.
    """

    # Reshape x to be a grid for use with the FNO
    n_batch = z.shape[0]
    input_dimension = x.shape[-1]
    n = round(x.shape[1] ** (1 / input_dimension))
    x_shape = [n_batch] + [n] * input_dimension + [x.shape[-1]]
    x = jnp.reshape(x, x_shape)

    # Lift z to be a constant function
    new_dims = x.ndim - 2
    z = jnp.reshape(z, [z.shape[0]] + [1] * new_dims + [z.shape[1]])
    tiling_shape = [1] + list(x.shape[1:-1]) + [1]
    z = jnp.tile(z, tiling_shape)

    # Concatenate the two functions on the channel axis
    zx = jnp.concatenate((z, x), axis=-1)

    # Apply the FNO
    u = operator(zx, x)

    # Reshape to the "sparse" convention
    u = jnp.reshape(u, (u.shape[0], -1, u.shape[-1]))
    return u
