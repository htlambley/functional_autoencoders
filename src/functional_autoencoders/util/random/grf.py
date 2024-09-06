import jax
import jax.numpy as jnp
import numpy as np

from functional_autoencoders.util.fft import idstn


def _index_to_frequency(idx, max):
    ret = np.zeros_like(idx)
    for pos, (i, m) in enumerate(zip(idx, max)):
        if i <= m // 2:
            ret[pos] = i
        else:
            ret[pos] = -(m - i)
    return ret


def _compute_torus_covariance_operator_sqrt_eigenvalues(shape, tau=3, d=2):
    eigs = np.zeros(shape)
    for index, _ in np.ndenumerate(eigs):
        idx = _index_to_frequency(index, shape)
        sum_square_idx = np.sum(np.square(idx))
        if sum_square_idx != 0:
            eigs[index] = (4 * np.pi**2 * sum_square_idx + tau**2) ** (-d / 2)
    return eigs


def torus_grf(key: jax.random.PRNGKey, n, shape, out_dim=1, tau=3, d=2, method="fft"):
    r"""Returns realisations of a mean-zero Gaussian random field on an $n$-dimensional torus with Matérn-type covariance operator and periodic boundary conditions.

    This function generates realisations of a mean-zero Gaussian random field on $X = L^{2}(\Omega; \mathbb{C})$, with
    $\Omega = \mathbb{T}^{n}$ defined as the domain $[0, 1]^{n}$ with periodic boundary.
    The covariance operator used is a perturbed inverse power of the periodic Laplacian on the torus,

    $$ C = (\tau^{2} I - \Delta)^{-d},$$

    which is trace class when $d > n/2$.
    The dimension $n$ is inferred from the dimensions of `shape`.

    ## Arguments
    `n` : int
        number of realisations to generate
    `shape` : tuple of int
        shape of each output realisation; output must be square if `method='fft'`, i.e. all values of `shape` equal.
    `tau` : float
        length-scale parameter
    `d` : float
        smoothness parameter
    `method`: 'fft'
        the method used to generate the Gaussian random field; currently only supports 'fft', which computes the eigenvalues
        of the covariance operator analytically and generates a truncated Karhunen--Lo\`eve expansion via the fast Fourier transform.

    ## Returns
    Array of shape `(n, *shape)` containing `n` realisations of the Gaussian random field with the required shape.
    Note that the returned random field is *complex-valued*, so users may wish to take only the real part, which gives a real-valued
    Gaussian random function with mean zero.
    """
    if method == "fft":
        if key is not None:
            key, subkey = jax.random.split(key)
            zhat = jax.random.normal(
                key, (n, *shape, out_dim)
            ) + 1j * jax.random.normal(subkey, (n, *shape, out_dim))
        else:
            zhat = np.random.randn(n, *shape, out_dim) + 1j * np.random.randn(
                n, *shape, out_dim
            )

        eigs = _compute_torus_covariance_operator_sqrt_eigenvalues(shape, tau=tau, d=d)
        eigs = jnp.expand_dims(eigs, -1)
        return jnp.fft.ifftn(eigs * zhat, norm="forward", axes=range(1, zhat.ndim - 1))
    else:
        raise NotImplementedError("Only supported generation method is method='fft'")


def _compute_dirichlet_covariance_operator_sqrt_eigenvalues(
    shape, tau=3, d=2, even_powers_only=False
):
    if len(shape) != 1 and even_powers_only:
        raise NotImplementedError("even_powers_only implemented only in 1D")
    eigs = np.zeros(shape)
    for index, _ in np.ndenumerate(eigs):
        sum_square_idx = np.sum(np.square(index + np.ones_like(index)))
        if sum_square_idx != 0:
            # Note here the 0th index is frequency 1, so the "even" frequencies are the odd indices
            if len(shape) == 1 and index[0] % 2 == 0 and even_powers_only:
                continue
            eigs[index] = (np.pi**2 * sum_square_idx + tau**2) ** (-d / 2) * 2 ** (
                -len(shape) / 2
            )
    return eigs


def dirichlet_grf(
    key: jax.random.PRNGKey,
    n,
    shape,
    out_dim=1,
    tau=3,
    d=2,
    method="dst",
    even_powers_only=False,
):
    """Returns realisations of a mean-zero Gaussian random field on the $n$-dimensional square $[0, 1]^{n}$ with Matérn-type covariance operator and zero Dirichlet boundary conditions.

    The arguments are similar to `torus_grf`, with the exception of `even_powers_only`, which assigns coefficient zero to all
    odd frequencies in the Karhunen–Loeve expansion for compatibility with the work of Seidman et al. (2023), as discussed below.

    ## Notes
    The variational autoencoding neural operator of Seidman et al. (2023) uses a dataset based on GRFs with Dirichlet Laplacian covariance.
    Their implementation subtly differs because they only allow even frequencies, as noticeable in Fig. 6 of their paper where all the generated
    functions pass through 0.5.
    """

    if method == "dst":
        if key is not None:
            zhat = jax.random.normal(key, (n, *shape, out_dim))
        else:
            zhat = np.random.randn(n, *shape, out_dim)

        eigs = _compute_dirichlet_covariance_operator_sqrt_eigenvalues(
            shape, tau=tau, d=d, even_powers_only=even_powers_only
        )
        eigs = jnp.expand_dims(eigs, -1)
        return idstn(eigs * zhat, norm="forward", axes=range(1, zhat.ndim - 1), type=1)
    else:
        raise NotImplementedError("Only supported generation method is method='kl'")
