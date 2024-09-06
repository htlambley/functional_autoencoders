from functional_autoencoders.datasets import GenerableDataset
from functional_autoencoders.util.random import grf
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import itertools


class GRF(GenerableDataset):
    r"""Realisations from a Gaussian process on [0, 1] with zero boundary conditions and Matérn-like covariance operator.

    The data are realisations from a Gaussian measure $N(0, C)$ on the space $L^{2}([0, 1])$ with the covariance operator given
    as a Matérn-like inverse power of the Dirichlet Laplacian:

    $$ C = (\tau^{2} I - \Delta)^{d}. $$

    The realisations are generated using the `dirichlet_grf` function in the `random` module, which uses a discrete sine transformation
    to efficiently generate realisations of the Gaussian process.

    Notes:
    This dataset supports transformations using transform (e.g. for rescaling). The VANO implementation of Seidman et al. (2023)
    instead applies a rescaling *in the ELBO loss* and this is preferable when trying to reproduce their results.

    References:
    Seidman et al. (2023). Variational autoencoding neural operator. ICML.
    """

    def __init__(
        self,
        samples=200,
        pts=2000,
        tau=3.0,
        d=2.0,
        even_powers_only=False,
        dim=1,
        out_dim=1,
        transform=None,
        *args,
        **kwargs,
    ):
        self._samples = samples
        self._pts = pts
        self._tau = tau
        self._d = d
        self._even_powers_only = even_powers_only
        self._dim = dim
        self._out_dim = out_dim
        self.transform = transform
        super().__init__(*args, **kwargs)

    def generate(self):
        if not self.train:
            raise NotImplementedError(
                "Test/train split not yet implemented; only train data available"
            )

        if self._samples <= 0 or self._pts <= 0:
            raise ValueError(
                "To generate `GRF` dataset, need number of realisations `samples > 0` and grid points `pts > 0`"
            )

        self._samples *= 2
        grid = [self._pts] * self._dim
        u = grf.dirichlet_grf(
            None,
            self._samples,
            grid,
            tau=self._tau,
            d=self._d,
            even_powers_only=self._even_powers_only,
            out_dim=self._out_dim,
        )
        x1 = np.linspace(0, 1, self._pts + 2)[1:-1]
        xs = np.meshgrid(*([x1] * self._dim), indexing="ij")
        xs = [np.expand_dims(v, -1) for v in xs]
        x = np.concatenate(xs, axis=-1)

        u = np.reshape(u, (u.shape[0], -1, self._out_dim))
        x = np.reshape(x, (-1, x.shape[-1]))

        self.data = {
            "u": u,
            "x": x,
            "d": self._d,
            "tau": self._tau,
            "even_powers_only": self._even_powers_only,
        }

    def __len__(self):
        return self.data["u"].shape[0] // 2

    def __getitem__(self, idx):
        if not self.train:
            idx += len(self)

        u = self.data["u"][idx]
        x = self.data["x"][:]
        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x

    @property
    def pts(self) -> float:
        return self.data["u"].shape[1]

    @property
    def tau(self) -> float:
        return self.data["tau"]

    @property
    def d(self) -> float:
        return self.data["d"]

    @property
    def x(self) -> np.array:
        return self.data["x"][:]

    @property
    def even_powers_only(self) -> bool:
        return self.data["even_powers_only"]

    @partial(jax.vmap, in_axes=(None, 0))
    def scaled_basis(self, x):
        if x.shape[-1] != 1 or self.data["u"].shape[-1] != 1:
            raise NotImplementedError()
        n = jnp.reshape(jnp.arange(1, x.shape[0] + 1), (1, -1))
        basis = 2 * jnp.sin(jnp.pi * n * x)
        sqrt_eigs = np.reshape(
            grf._compute_dirichlet_covariance_operator_sqrt_eigenvalues(
                (x.shape[0],),
                tau=self.tau,
                d=self.d,
                even_powers_only=self.even_powers_only,
            ),
            (1, -1),
        )
        return sqrt_eigs * basis


class GaussianDensities(GenerableDataset):
    def __init__(
        self,
        samples=2048,
        pts=48,
        n_gaussians=1,
        std_min=0.01,
        std_max=0.11,
        transform=None,
        *args,
        **kwargs,
    ):
        self._samples = samples
        self._pts = pts
        self._n_gaussians = n_gaussians
        self._std_min = std_min
        self._std_max = std_max
        self.transform = transform
        super().__init__(*args, **kwargs)

    def generate(self):
        if self._samples <= 0 or self._pts <= 0:
            raise ValueError(
                "To generate `GaussianDensities` dataset, need number of realisations `samples > 0` and grid points `pts > 0`"
            )

        # Generate equal test--train split (i.e. _samples of train and _samples of test)
        self._samples *= 2
        means = np.random.uniform(size=(self._samples, self._n_gaussians, 2))
        stds = np.random.uniform(
            low=self._std_min,
            high=self._std_max,
            size=(self._samples, self._n_gaussians),
        )
        x = np.linspace(0, 1, self._pts + 1)[:-1]
        xs = np.array(list(itertools.product(x, x)))
        u = np.zeros((self._samples, self._pts, self._pts))
        for sample in range(0, self._samples):
            for i in range(0, self._n_gaussians):
                out = np.reshape(
                    scipy.stats.multivariate_normal.pdf(
                        xs,
                        mean=means[sample, i, :],
                        cov=((stds[sample, i]) ** 2) * np.eye(2),
                    ),
                    (self._pts, self._pts),
                )
                u[sample, :, :] += out
        u = u.reshape(self._samples, -1, 1)

        self.data = {"u": u, "x": x}

    def __len__(self):
        return self.data["u"].shape[0] // 2

    def __getitem__(self, idx):
        if not self.train:
            idx += len(self)
        u = self.data["u"][idx]
        x = self.data["x"][:]
        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x

    @property
    def x(self) -> np.array:
        return self.data["x"][:]
