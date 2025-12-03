import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from functools import partial
from typing import Callable, Sequence
from functional_autoencoders.util.pca import pca
from functional_autoencoders.domains import Domain
from functional_autoencoders.util.random.sde import add_bm_noise
from functional_autoencoders.util import bucket_data, get_transition_matrix
from functional_autoencoders.samplers.sampler_vae import SamplerVAE
from functional_autoencoders.autoencoder import Autoencoder


NormFunction = Callable[[ArrayLike, ArrayLike], jax.Array]
MMDKernel = Callable[[ArrayLike, ArrayLike, ArrayLike], jax.Array]


def squared_l2_norm(u: ArrayLike, x: ArrayLike) -> jax.Array:
    return jnp.mean(jnp.square(u), axis=(1, 2))


def functional_squared_exponential(
    sigma: float, norm: NormFunction = squared_l2_norm
) -> MMDKernel:
    """
    Returns a functional squared exponential kernel.

    Arguments:

    u : jnp.array of shape [samples, n_evals, out_dim]
    v : jnp.array of shape [samples, n_evals, out_dim]
    x : jnp.array of shape [n_evals, in_dim]
    """

    def inner(u: ArrayLike, v: ArrayLike, x: ArrayLike) -> jax.Array:
        diff = u - v
        squared_l2_norm = norm(diff, x)
        return jnp.exp(-squared_l2_norm / (2 * sigma**2))

    return inner


def maximum_mean_discrepancy(
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike,
    kernel: MMDKernel = functional_squared_exponential(sigma=1),
) -> jax.Array:
    """Computes a biased empirical estimate of the maximum mean discrepancy between the functional datasets `u` and `v` with the given kernel.

    This implements the biased empirical maximum mean discrepancy (MMD) estimator given by equation (5) of Gretton et al. (2012).

    Arguments:
    u : jnp.array of shape [samples, n_evals, out_dim]
    v : jnp.array of shape [samples, n_evals, out_dim]
    x : jnp.array of shape [n_evals, in_dim]

    References:

    Gretton et al. (2012). *A kernel two-sample test*. JMLR 13:723--773.
    """

    u1 = jnp.repeat(u, u.shape[0], axis=0)
    u2 = jnp.tile(u, (u.shape[0], 1, 1))

    v1 = jnp.repeat(v, v.shape[0], axis=0)
    v2 = jnp.tile(v, (v.shape[0], 1, 1))

    uv1 = jnp.repeat(u, v.shape[0], axis=0)
    uv2 = jnp.tile(v, (u.shape[0], 1, 1))

    u_terms = jnp.mean(kernel(u1, u2, x))
    uv_terms = 2 * jnp.mean(kernel(uv1, uv2, x))
    v_terms = jnp.mean(kernel(v1, v2, x))
    # Occasionally the sum will be below zero because of numerical errors;
    # to avoid propagating nans, we take the max of the sum and zero
    sum = u_terms - uv_terms + v_terms
    sum = jnp.where(sum >= 0, sum, 0.0)
    return sum**0.5


def generalised_maximum_mean_discrepancy(
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike,
    kernels: Sequence[MMDKernel],
) -> jax.Array:
    """Computes a biased empirical estimate of the generalised maximum mean discrepancy between the functional datasets `u` and `v` given a collection of kernels.

    This implements the generalised maximum mean discrepancy (MMD) given in equation (6) of Sriperumbudur et al. (2009).

    References:
    Sriperumbudur et al. (2009). *Kernel choice and classifiability for RKHS embeddings of probability distributions*. NeurIPS.
    """
    return jnp.max(
        jnp.array([maximum_mean_discrepancy(u, v, x, kernel) for kernel in kernels])
    )


class Metric:
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def batched(self) -> bool:
        raise NotImplementedError()

    def __call__(self, state, key, test_dataloader):
        if self.batched:
            metric_value = 0.0
            for i, batch in enumerate(test_dataloader):
                key, subkey = jax.random.split(key)
                metric_value += self.call_batched(state, batch, subkey)
            return metric_value / (i + 1)
        else:
            key, subkey = jax.random.split(key)
            return self.call_unbatched(state, test_dataloader, subkey)


class MSEMetric(Metric):
    def __init__(self, autoencoder: Autoencoder, domain: Domain):
        self.autoencoder = autoencoder
        self.domain = domain

    @property
    def batched(self) -> bool:
        return True

    @property
    def name(self) -> str:
        norm = self.domain.name
        return f"MSE (in {norm})"

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u, x, _, _ = batch

        vars = {"params": state.params, "batch_stats": state.batch_stats}
        v_hat = self.autoencoder.apply(vars, u, x, x, train=False)
        return jnp.mean(self.domain.squared_norm(u - v_hat, x))


class BatchedMMDMetric(Metric):
    def __init__(self, model, kernels, latent_dim):
        self.model = model
        self.kernels = kernels
        self.latent_dim = latent_dim

    @property
    def batched(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "Batched MMD"

    @partial(jax.jit, static_argnums=(0, 3))
    def _sample(self, variables, x, batch_size, key):
        latents = jax.random.normal(key, (batch_size, self.latent_dim))
        output = self.model.decode(variables, latents, x)
        return output

    @partial(jax.jit, static_argnums=0)
    def _mmd(self, trues, samples, x):
        return generalised_maximum_mean_discrepancy(trues, samples, x, self.kernels)

    def call_batched(self, state, u, x, key):
        samples = self._sample(state, x, u.shape[0], key)
        return self._mmd(u, samples, x)


class MMDMetric(Metric):
    def __init__(self, model, kernels, latent_dim, batch_size, n_samples=512):
        self.model = model
        self.kernels = kernels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    @property
    def batched(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "MMD"

    @partial(jax.jit, static_argnums=0)
    def _sample(self, state, x, key):
        latents = jax.random.normal(key, (self.batch_size, self.latent_dim))
        output = self.model.decode(state, latents, x, train=False)
        return output

    @partial(jax.jit, static_argnums=0)
    def _mmd(self, trues, samples, x):
        return generalised_maximum_mean_discrepancy(trues, samples, x, self.kernels)

    def call_unbatched(self, state, test_dataloader, key):
        # determine output shape
        key, subkey = jax.random.split(key)
        _, x, _, _ = next(iter(test_dataloader))
        shape = self._sample(state, x, subkey).shape

        # sample n_samples from the true and the generative model
        n = 0
        samples = np.zeros((self.n_samples, *shape[1:]))
        trues = np.zeros_like(samples)
        for u, x, _, _ in test_dataloader:
            key, subkey = jax.random.split(key)
            u_samples = self._sample(state, x, subkey)
            samples[n : n + self.batch_size, :, :] = u_samples
            trues[n : n + self.batch_size, :, :] = u
            n += self.batch_size

            if n == self.n_samples:
                break
            elif n > self.n_samples:
                raise ValueError(
                    "Requested number of samples not evenly divisible by batch size"
                )
        if n < self.n_samples:
            raise ValueError(
                "Insufficient entries in test dataset. Reduce `n_samples` in `MMDMetric`"
            )

        # Only the first x in the batch is used for the MMD calculation
        return self._mmd(trues, samples, x[0])


class TransitionMatrixDiffMetric(Metric):
    def __init__(self, autoencoder, config_data, theta, x_locs, y_locs):
        self.autoencoder = autoencoder
        self.config_data = config_data
        self.theta = theta
        self.x_locs = x_locs
        self.y_locs = y_locs

    @property
    def name(self) -> str:
        return "Transition Matrix Diff"

    @property
    def batched(self) -> bool:
        return False

    def call_unbatched(self, state, test_dataloader, key):
        u_test_short = jnp.array(test_dataloader.dataset.data["u"][:1000])

        key, subkey = jax.random.split(key)
        t_batch = jnp.repeat(
            test_dataloader.dataset.x[None, ...], u_test_short.shape[0], axis=0
        )

        key, subkey = jax.random.split(key)
        sampler = SamplerVAE(self.autoencoder, state)
        u_samples = sampler.sample(t_batch, subkey)

        u_samples = add_bm_noise(
            samples=u_samples,
            epsilon=self.config_data["epsilon"],
            theta=self.theta,
            sim_dt=self.config_data["sim_dt"],
            T=self.config_data["T"],
        )

        n_buckets = (len(self.x_locs) - 1) * (len(self.y_locs) - 1)

        u_bucket_true_short = bucket_data(u_test_short, self.x_locs, self.y_locs)
        u_bucket_sample = bucket_data(u_samples, self.x_locs, self.y_locs)

        transition_matrix_true_short = get_transition_matrix(
            u_bucket_true_short, n_buckets
        ).mean(0)
        transition_matrix_samples = get_transition_matrix(
            u_bucket_sample, n_buckets
        ).mean(0)

        trans_mat_short_vs_samples = jnp.linalg.norm(
            transition_matrix_true_short - transition_matrix_samples
        )

        return trans_mat_short_vs_samples


class PCAHSMetric(Metric):
    def __init__(self, model, n_samples=512, cutoff_freq=None):
        self.model = model
        self.n_samples = n_samples
        self.cutoff_freq = cutoff_freq

    @property
    def name(self) -> str:
        return "Avg PCA Hilbert--Schmidt rel. error"

    @property
    def batched(self) -> bool:
        return False

    @partial(jax.jit, static_argnums=0)
    def _pca(self, samples):
        if samples.shape[-1] > 1:
            raise NotImplementedError(
                "`hilbert_schmidt` does not support out_dim > 1 yet."
            )
        eigenvalues, eigenvectors = pca(samples)
        return jnp.sqrt(jnp.abs(eigenvalues)) * eigenvectors

    def _hs(self, eigenvectors_pred, eigenvectors_true):
        # Form empirical covariance matrices
        eigenvectors_pred = jnp.expand_dims(eigenvectors_pred.T, 1)
        eigenvectors_true = jnp.expand_dims(eigenvectors_true.T, 1)
        cov_pred = jnp.sum(
            jnp.swapaxes(eigenvectors_pred, -1, -2) @ eigenvectors_pred, axis=0
        )
        cov_true = jnp.sum(
            jnp.swapaxes(eigenvectors_true, -1, -2) @ eigenvectors_true, axis=0
        )

        # Compute relative Hilbert--Schmidt norm error.
        numerator = jnp.sum(jnp.square(cov_pred - cov_true)) ** (0.5)
        denominator = jnp.sum(jnp.square(cov_true)) ** (0.5)
        return float(numerator / denominator)

    def call_unbatched(self, state, test_dataloader, key):
        # Only supports variational autoencoders but could
        # be extended to support deterministic autoencoders.
        assert self.model.encoder.is_variational

        _, x = next(iter(test_dataloader))
        x_single = jnp.expand_dims(x[0], 0)

        tiling_shape = [self.n_samples] + [1] * (x_single.ndim - 1)
        x_tile = jnp.tile(x_single, tiling_shape)

        key, subkey = jax.random.split(key)
        sampler = SamplerVAE(self.model, state)
        samples = sampler.sample(x_tile, subkey)

        pred_scaled_basis = self._pca(samples)[:, :, : self.cutoff_freq]

        scaled_basis = test_dataloader.dataset.scaled_basis(x_single)[0]
        return self._hs(pred_scaled_basis, scaled_basis)
