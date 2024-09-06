import sys

sys.path.append(".")
import os
import jax
import jax.numpy as jnp
from typing import Literal
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


def potential_2d(x):
    parabola = lambda x: jnp.square(x).sum(axis=-1)
    linear = lambda x: jnp.sum(x, axis=-1)
    neg_gaussian = lambda x, C, sigma: -jnp.prod(
        jax.scipy.stats.norm.pdf(x.reshape(-1, 2), loc=C, scale=sigma), axis=-1
    ).reshape(x.shape[:-1])

    mu_list = [
        [0, 0],
        [0.2, 0.2],
        [-0.2, -0.2],
        [0.2, -0.2],
        [0, 0.2],
        [-0.2, 0],
    ]

    sigma_list = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.03,
        0.03,
    ]

    coeff_list = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.01,
        0.01,
    ]

    p = parabola(x)
    lin = linear(x)
    result = p + 0.5 * lin

    for mu, sigma, coeff in zip(mu_list, sigma_list, coeff_list):
        result += coeff * neg_gaussian(x, jnp.array(mu), sigma)

    return 0.3 * result


def get_sde_dataloaders(
    config_data, verbose, samples=None, which: Literal["train", "test", "both"] = "both"
):
    drift = get_brownian_dynamics_drift(potential_2d)
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


def run_sde2d(key, output_dir, config_path, verbose=True):
    config_sde2d = yaml_load(config_path)
    config_data = config_sde2d["data"]

    train_dataloader, test_dataloader = get_sde_dataloaders(config_data, verbose)
    key, subkey = jax.random.split(key)
    trainer = get_trainer(subkey, config_sde2d, train_dataloader, test_dataloader)

    key, subkey = jax.random.split(key)
    results = fit_trainer_using_config(
        subkey, trainer, config_sde2d, verbose="metrics" if verbose else "none"
    )

    save_model_results(
        autoencoder=trainer.autoencoder,
        results=results,
        model_dir=os.path.join(output_dir, "models"),
    )
