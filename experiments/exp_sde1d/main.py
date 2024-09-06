import sys

sys.path.append(".")
import os
import jax
from typing import Literal
from jax.typing import ArrayLike
from experiments.trainer_loader import get_trainer
from functional_autoencoders.datasets import get_dataloaders, RandomMissingData
from functional_autoencoders.datasets.sde import (
    SDE,
    get_brownian_dynamics_diffusion,
    get_brownian_dynamics_drift,
)
from functional_autoencoders.util import (
    save_model_results,
    yaml_load,
    fit_trainer_using_config,
)


def potential_1d(x: ArrayLike, c: float = 0.5, alpha: float = 12):
    r"""
    The potential

    $$U(x) = \alpha\left( \frac{1}{4} x^{4} + \frac{c}{3} x^{3} - \frac{1}{2} x^{2} - c x \right),$$

    which, with the particular choices $c = 1/2$ and $\alpha = 12$, yields the potential (2.26) in
    [the paper](https://arxiv.org/pdf/2408.01362), with minima at $-1$ and $+1$.

    :param x: `ArrayLike` of shape `[1]`.
    :param c: `float` parameter for potential $U$ (see above).
    :param alpha: `float` parameter for potential $U$ (see above).
    """
    return alpha * ((c / 3) * x[0] ** 3 - c * x[0] + 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2)


def get_sde_dataloaders(
    config_data, verbose, samples=None, which: Literal["train", "test", "both"] = "both"
):
    drift = get_brownian_dynamics_drift(potential_1d)
    diffusion = get_brownian_dynamics_diffusion(config_data["epsilon"])
    point_ratio_train = config_data["point_ratio_train"]
    random_missing_data = RandomMissingData(point_ratio_train)

    if which == "train" or which == "both":
        train_dataloader = get_dataloaders(
            SDE,
            drift=drift,
            diffusion=diffusion,
            T=config_data["T"],
            samples=config_data["samples"] if samples is None else samples,
            pts=config_data["pts"],
            sim_dt=config_data["sim_dt"],
            batch_size=config_data["batch_size"],
            num_workers=config_data["num_workers"],
            x0=config_data["x0"],
            transform_generated=random_missing_data,
            which="train",
            verbose=verbose,
        )
    else:
        train_dataloader = None

    if which == "test" or which == "both":
        test_dataloader = get_dataloaders(
            SDE,
            drift=drift,
            diffusion=diffusion,
            T=config_data["T"],
            samples=config_data["samples"] if samples is None else samples,
            pts=config_data["pts"],
            sim_dt=config_data["sim_dt"],
            batch_size=config_data["batch_size"],
            num_workers=config_data["num_workers"],
            x0=config_data["x0"],
            which="test",
            verbose=verbose,
        )
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader


def run_sde1d(key, output_dir, config_path, theta_list, verbose=True):
    config_sde1d = yaml_load(config_path)
    config_data = config_sde1d["data"]
    train_dataloader, test_dataloader = get_sde_dataloaders(config_data, verbose)

    for theta in theta_list:
        config_sde1d["loss"]["options"]["fvae_sde"]["theta"] = theta

        key, subkey = jax.random.split(key)
        trainer = get_trainer(subkey, config_sde1d, train_dataloader, test_dataloader)

        key, subkey = jax.random.split(key)
        results = fit_trainer_using_config(
            subkey, trainer, config_sde1d, verbose="metrics" if verbose else "none"
        )

        save_model_results(
            autoencoder=trainer.autoencoder,
            results=results,
            model_dir=os.path.join(output_dir, "models", str(theta)),
        )
