import sys

sys.path.append(".")
import os
import jax
from functional_autoencoders.datasets import get_dataloaders, ComplementMasking
from functional_autoencoders.datasets.navier_stokes import NavierStokes
from functional_autoencoders.datasets.darcy_flow import DarcyFlow
from experiments.trainer_loader import get_trainer
from functional_autoencoders.util import (
    save_model_results,
    yaml_load,
    fit_trainer_using_config,
)


def run_sparse_training(
    key,
    output_dir,
    config_path,
    ratio_rand_pts_enc,
    ns_viscosity,
    is_darcy=False,
    verbose="metrics",
):

    config = yaml_load(config_path)

    mask_train = ComplementMasking(ratio_rand_pts_enc)
    mask_test = ComplementMasking(ratio_rand_pts_enc)
    if not is_darcy:
        train_dataloader, test_dataloader = get_dataloaders(
            NavierStokes,
            data_base=".",
            viscosity=ns_viscosity,
            transform_train=mask_train,
            transform_test=mask_test,
        )
    else:
        train_dataloader, test_dataloader = get_dataloaders(
            DarcyFlow,
            data_base=".",
            transform_train=mask_train,
            transform_test=mask_test,
            downscale=9,
        )

    key, subkey = jax.random.split(key)
    trainer = get_trainer(subkey, config, train_dataloader, test_dataloader)

    key, subkey = jax.random.split(key)
    results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

    save_model_results(
        autoencoder=trainer.autoencoder,
        results=results,
        model_dir=os.path.join(output_dir, "models"),
    )
