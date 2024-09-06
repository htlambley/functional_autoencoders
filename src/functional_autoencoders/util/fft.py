import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Literal


def _invert_norm(
    norm: Literal["backward", "ortho", "forward"]
) -> Literal["backward", "ortho", "forward"]:
    if norm == "backward":
        return "forward"
    elif norm == "forward":
        return "backward"
    return norm


def dst(
    x: ArrayLike,
    type: Literal[1, 2, 3, 4] = 2,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> jax.Array:
    """
    LAX-backed implementation of the discrete sine transform (DST), following the `scipy.fft.dst` API and its defaults.

    Notes:
    - Currently only implements DST-I, emulating `scipy.fft.dst` with `type = 1`.
    - The algorithm used is a simple implementation of the DST based on the discrete Fourier transform (DFT), which
      forms an intermediate tensor of double the input size.

    ## References
    Press, Teukolsky, Vetterling & Flannery (2007). Numerical recipes in C: the art of scientific computing.
        3rd ed. Cambridge University Press. ISBN: 9780521880688.
    """
    if type != 1:
        raise NotImplementedError()

    norm = _invert_norm(norm)
    shape = list(x.shape)
    shape[axis] = 1
    xaug = jnp.concatenate(
        [
            jnp.zeros(shape),
            x,
            jnp.zeros(shape),
            -jnp.flip(x, axis=axis),
        ],
        axis=axis,
    )
    xhat = jnp.fft.ifft(xaug, norm=norm, axis=axis)
    idx = [slice(0, dim) for dim in x.shape]
    idx[axis] = slice(1, x.shape[axis] + 1)
    return (xhat.imag)[tuple(idx)]


def dstn(
    x: ArrayLike,
    type: Literal[1, 2, 3, 4] = 2,
    axes=None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> jax.Array:
    """
    LAX-backed implementation of the $n$-dimensional discrete sine transform (DST), emulating the `scipy.fft.dstn` API.

    The implementation is based on repeated application of `dst` in each relevant axis (all axes by default, unless `axes` is
    specified).

    ## References
    Press, Teukolsky, Vetterling & Flannery (2007). Numerical recipes in C: the art of scientific computing.
        3rd ed. Cambridge University Press. ISBN: 9780521880688.
    """
    if type != 1:
        raise NotImplementedError()

    if axes is None:
        axes = range(0, x.ndim)

    for axis in axes:
        x = dst(x, type=type, axis=axis, norm=norm)
    return x


def idst(
    x: ArrayLike,
    type: Literal[1, 2, 3, 4] = 2,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> jax.Array:
    """
    LAX-backed implementation of the inverse discrete sine transform (DST) emulating the `scipy.fft.idst` API.

    ## References
    Press, Teukolsky, Vetterling & Flannery (2007). Numerical recipes in C: the art of scientific computing.
        3rd ed. Cambridge University Press. ISBN: 9780521880688.
    """
    norm = _invert_norm(norm)
    return dst(x, type, axis, norm)


def idstn(
    x: ArrayLike,
    type: Literal[1, 2, 3, 4] = 2,
    axes=None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> jax.Array:
    """
    LAX-backed implementation of the $n$-dimensional inverse discrete sine transform (DST),
    emulating the `scipy.fft.idstn` API.

    ## References
    Press, Teukolsky, Vetterling & Flannery (2007). Numerical recipes in C: the art of scientific computing.
        3rd ed. Cambridge University Press. ISBN: 9780521880688.
    """
    norm = _invert_norm(norm)
    return dstn(x, type, axes, norm)
