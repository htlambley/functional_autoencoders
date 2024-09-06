import numpy as np
import jax
from functional_autoencoders.util.random.sde import euler_maruyama
from functional_autoencoders.datasets import GenerableDataset


def get_brownian_dynamics_drift(potential, *args, **kwargs):
    neg_pot = lambda x: -potential(x, *args, **kwargs)
    net_pot_grad = jax.grad(neg_pot)
    neg_pot_grad_vmap = jax.vmap(net_pot_grad, in_axes=(0,))
    neg_pot_grad_vmap_jit = jax.jit(neg_pot_grad_vmap)
    return lambda X, t: neg_pot_grad_vmap_jit(X)


def get_brownian_dynamics_diffusion(epsilon):
    return lambda x, dwt, t: (epsilon ** (0.5)) * dwt


class SDE(GenerableDataset):
    def __init__(
        self,
        drift,
        diffusion,
        x0,
        samples=200,
        pts=100,
        T=1,
        sim_dt=1e-3,
        verbose=False,
        transform=None,
        transform_generated=None,
        *args,
        **kwargs,
    ):
        self._samples = samples
        self._pts = pts
        self._sim_dt = sim_dt
        self._dt = T / pts
        self._subsample_rate = int(self._dt / sim_dt)
        self._n_steps = pts * self._subsample_rate
        self.drift = drift
        self.diffusion = diffusion
        self.verbose = verbose
        self._x0 = x0
        self.transform = transform
        self.transform_generated = transform_generated
        super().__init__(*args, **kwargs)

    def generate(self):
        if self._samples <= 0 or self._pts <= 0:
            raise ValueError(
                "To generate dataset, need number of realisations `samples > 0` and grid points `pts > 0`"
            )

        x0 = np.repeat(np.expand_dims(self._x0, 0), self._samples, axis=0)
        u = euler_maruyama(
            x0,
            self.drift,
            self.diffusion,
            self._sim_dt,
            self._n_steps,
            self._subsample_rate,
            self.verbose,
        )

        x = np.arange(0, self._pts + 1) * self._dt
        x = np.expand_dims(x, -1)

        if self.transform_generated is not None:
            u, x = self.transform_generated(u, x)

        self.data = {
            "u": u,
            "x": x,
            "x0": x0,
            "sim_dt": self._sim_dt,
            "samples": self._samples,
        }

    def __len__(self):
        return self.data["samples"]

    def __getitem__(self, idx):
        u = self.data["u"][idx]
        x = self.data["x"][:]
        if self.transform is not None:
            return self.transform(u, x)
        else:
            return u, x, u, x

    @property
    def x0(self):
        return self.data["x0"][:]

    @property
    def x(self):
        return self.data["x"][:]

    @property
    def sim_dt(self) -> float:
        return self.data["sim_dt"]
