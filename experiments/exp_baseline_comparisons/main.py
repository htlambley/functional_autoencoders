import sys

sys.path.append(".")
import os
import jax
from functional_autoencoders.datasets import get_dataloaders
from functional_autoencoders.datasets.navier_stokes import NavierStokes
from functional_autoencoders.datasets.darcy_flow import DarcyFlow
from functional_autoencoders.util import (
    save_data_results,
    save_model_results,
    yaml_load,
    fit_trainer_using_config,
)
from experiments.trainer_loader import get_trainer


def run_baseline_comparisons(
    key,
    output_dir,
    config_path,
    n_runs,
    ns_viscosity,
    is_darcy=False,
    verbose="metrics",
):

    config = yaml_load(config_path)

    if not is_darcy:
        train_dataloader, test_dataloader = get_dataloaders(
            NavierStokes,
            data_base=".",
            viscosity=ns_viscosity,
        )
    else:
        train_dataloader, test_dataloader = get_dataloaders(
            DarcyFlow, data_base=".", downscale=9
        )

    for run_idx in range(n_runs):
        key, subkey = jax.random.split(key)
        trainer = get_trainer(subkey, config, train_dataloader, test_dataloader)

        key, subkey = jax.random.split(key)
        results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

        save_model_results(
            autoencoder=trainer.autoencoder,
            results=results,
            model_dir=os.path.join(output_dir, "models", str(run_idx)),
        )

        save_data_results(
            autoencoder=trainer.autoencoder,
            results=results,
            test_dataloader=test_dataloader,
            data_dir=os.path.join(output_dir, "data", str(run_idx)),
        )
