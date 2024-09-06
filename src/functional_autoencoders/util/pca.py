import jax.numpy as jnp


def pca(u):
    r"""
    Performs principal component analysis for discretisations of functional data $(u_{j})_{j = 1}^{N} \subset L^{2}$
    by computing eigenfunctions and eigenvaleus of the empirical covariance operator

    $$C_{N} = \frac{1}{N} \sum_{j = 1}^{N} u_{j} \otimes u_{j}.$$

    Returns :math:`L^{2}`-orthonormal eigenvectors and associated eigenvalues of the empirical covariance operator.

    Note: this currently only works for functions taking values in :math:`\mathbb{R}^{d}`.

    :param u: `jnp.array` of shape `[samples, grid_pts, out_dim]`

    Returns tuple of:
    - **eigenvalues**: `jnp.array` of shape `[n_eigs]`
        sorted from smallest to largest
    - **eigenvectors**: `jnp.array` of shape `[grid_points, out_dim, n_eigs]`.


    ## References
    Bhattacharya, Hosseini, Kovachki, Stuart (2021). Model reduction and neural networks for parametric PDEs.
      SMAI J. Comp. Math 7:121--157, doi:[10.5802/smai-jcm.74](https://dx.doi.org/10.5802/smai-jcm.74).
    """

    grid_pts = u.shape[1]
    out_dim = u.shape[2]
    u = jnp.reshape(u, (u.shape[0], -1, 1))

    # Form empirical covariance operator
    cov = (1.0 / u.shape[0]) * jnp.sum(u @ jnp.conj(jnp.swapaxes(u, -1, -2)), axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Normalise eigenvalues and eigenvectors to account for L^2 inner product instead of Euclidean
    eigenvectors = eigenvectors.real
    l2_norms = jnp.expand_dims(
        jnp.mean(jnp.abs(eigenvectors) ** 2, axis=0) ** (-0.5), 0
    )
    normalised_eigenvectors = eigenvectors * l2_norms
    normalised_eigenvalues = eigenvalues.real / u.shape[1]
    normalised_eigenvectors = jnp.reshape(
        normalised_eigenvectors, (grid_pts, out_dim, -1)
    )
    return normalised_eigenvalues, normalised_eigenvectors
