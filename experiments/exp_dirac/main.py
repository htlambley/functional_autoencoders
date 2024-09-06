import sys

sys.path.append(".")
import os
import jax
from functional_autoencoders.datasets import get_dataloaders
from functional_autoencoders.datasets.dirac import RandomDirac
from experiments.trainer_loader import get_trainer
from functional_autoencoders.util import (
    save_data_results,
    yaml_load,
    fit_trainer_using_config,
)


def run_dirac(key, output_dir, config_path, n_runs, resolutions, verbose="metrics"):
    config = yaml_load(config_path)

    for resolution in resolutions:
        for run_idx in range(n_runs):
            train_dataloader, test_dataloader = get_dataloaders(
                RandomDirac,
                pts=resolution,
                fixed_centre=False,
                batch_size=1,
            )

            key, subkey = jax.random.split(key)
            trainer = get_trainer(subkey, config, train_dataloader, test_dataloader)

            key, subkey = jax.random.split(key)
            results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

            save_data_results(
                autoencoder=trainer.autoencoder,
                results=results,
                test_dataloader=test_dataloader,
                data_dir=os.path.join(
                    output_dir, "data", str(resolution), str(run_idx)
                ),
            )
