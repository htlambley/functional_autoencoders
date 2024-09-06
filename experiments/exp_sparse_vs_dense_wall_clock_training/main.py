import sys

sys.path.append(".")
import os
import jax
from time import time
from functional_autoencoders.datasets import get_dataloaders, RandomMasking
from functional_autoencoders.datasets.darcy_flow import DarcyFlow
from experiments.trainer_loader import get_trainer
from functional_autoencoders.util import (
    save_data_results,
    save_model_results,
    yaml_load,
    fit_trainer_using_config,
)


def run_sparse_vs_dense_wall_clock_training(
    key,
    output_dir,
    config_path,
    n_runs,
    downscale,
    ratio_rand_pts_enc_train_list,
    verbose="metrics",
):

    config = yaml_load(config_path)

    _, test_dataloader_full = get_dataloaders(
        DarcyFlow, data_base=".", downscale=downscale
    )

    for run_idx in range(n_runs):
        for ratio_rand_pts_enc_train in ratio_rand_pts_enc_train_list:
            mask_train = RandomMasking(
                ratio_rand_pts_enc_train, ratio_rand_pts_enc_train
            )
            train_dataloader, _ = get_dataloaders(
                DarcyFlow,
                data_base=".",
                transform_train=mask_train,
                downscale=downscale,
            )

            key, subkey = jax.random.split(key)
            trainer = get_trainer(
                subkey, config, train_dataloader, test_dataloader_full
            )

            start_time = time()

            key, subkey = jax.random.split(key)
            results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

            training_time = time() - start_time

            save_model_results(
                autoencoder=trainer.autoencoder,
                results=results,
                model_dir=os.path.join(
                    output_dir, "models", str(run_idx), str(ratio_rand_pts_enc_train)
                ),
            )

            save_data_results(
                autoencoder=trainer.autoencoder,
                results=results,
                test_dataloader=test_dataloader_full,
                data_dir=os.path.join(
                    output_dir, "data", str(run_idx), str(ratio_rand_pts_enc_train)
                ),
                additional_data={
                    "train_point_ratio": ratio_rand_pts_enc_train,
                    "training_time": training_time,
                },
            )
