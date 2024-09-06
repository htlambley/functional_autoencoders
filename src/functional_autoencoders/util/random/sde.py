import numpy as np
from typing import Callable
from tqdm.auto import tqdm


Drift = Callable[[np.array, float], np.array]
Diffusion = Callable[[np.array, np.array, float], np.array]


def euler_maruyama(
    x0: np.array,
    drift: Drift,
    diffusion: Diffusion,
    simulation_dt: float,
    n_steps: int,
    subsample_rate: int = 1,
    verbose: bool = False,
):
    r"""
    Simulates a stochastic differential equation of the form
    .. math::
        \mathrm{d} X_{t} = a(X_{t}, t}) \mathrm{d} t + b(X_{t}, t) \mathrm{d} W_{t}

    using the Euler--Maruyama scheme.

    Arguments:
    x0 : np.array of shape [n_realisations, dimension]
        The initial conditions for the SDE. The size of the leading axis determines the number of
        realisations to simulate.
    drift : function (x: [n_realisations, dimension], t: float) -> [n_realisations, dimension]
        Drift function :math:`a(X_{t}, t)` for the SDE, which should be batched over the leading axis.
    diffusion : function (x: [n_realisations, dimension], dwt: [n_realisations, dimension], t: float) -> [n_realisations, dimension]
        Diffusion function :math:`b(X_{t}, t)` for the SDE, which should be batched over the leading axis.
    simulation_dt : float
        Timestep for internal Euler--Maruyama solver
    n_steps : int
        Number of time steps to take. This determines the final time by T = dt * n_steps.
    subsample_rate : int
        Number of simulation steps per step reported in the output array. The dt in the output array is thus
        simulation_dt * subsample_rate.
        n_steps must be divisible by subsample_rate.

    Returns an array of shape [n_realisations, dimension, n_steps + 1], including the initial condition.

    The Euler--Maruyama scheme converges at rate :math:`\sqrt{t}` to the true solution.
    """
    n_realisations = x0.shape[0]
    d = x0.shape[1]
    x = x0
    result = np.zeros((n_realisations, n_steps // subsample_rate + 1, d))
    result[:, 0, :] = x0
    t = 0.0
    for i in tqdm(range(0, n_steps), disable=not verbose):
        dwt = np.sqrt(simulation_dt) * np.random.randn(n_realisations, d)
        x = x + drift(x, t) * simulation_dt + diffusion(x, dwt, t)
        if i % subsample_rate == 0:
            result[:, i // subsample_rate + 1, :] = x
        t = t + simulation_dt
    return result


def add_bm_noise(samples, epsilon, theta, sim_dt, T):
    n_batch, n_pts, n_dim = samples.shape
    x0 = np.zeros((n_batch, n_dim))
    bm = euler_maruyama(
        x0,
        lambda xt, t: -theta * (xt),
        lambda xt, dwt, t: (epsilon ** (0.5)) * dwt,
        sim_dt,
        int(T / sim_dt),
        int((T / (n_pts - 1)) / sim_dt),
    )
    return samples + bm
