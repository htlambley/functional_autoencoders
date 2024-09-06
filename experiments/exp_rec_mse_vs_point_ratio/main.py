import sys

sys.path.append(".")
import os
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from functional_autoencoders.datasets import get_dataloaders, ComplementMasking
from functional_autoencoders.datasets.navier_stokes import NavierStokes
from experiments.trainer_loader import get_trainer
from functional_autoencoders.util import (
    save_data_results,
    save_model_results,
    yaml_load,
    fit_trainer_using_config,
)


def run_rec_mse_vs_point_ratio(
    key,
    output_dir,
    config_path,
    n_runs,
    ns_viscosity,
    enc_point_ratio_train_list,
    enc_point_ratio_test_list,
    verbose="metrics",
):

    config = yaml_load(config_path)

    for run_idx in range(n_runs):
        for enc_point_ratio_train in enc_point_ratio_train_list:
            mask_train = ComplementMasking(enc_point_ratio_train)
            train_dataloader, test_dataloader = get_dataloaders(
                NavierStokes,
                data_base=".",
                viscosity=ns_viscosity,
                transform_train=mask_train,
            )

            key, subkey = jax.random.split(key)
            trainer = get_trainer(subkey, config, train_dataloader, test_dataloader)

            key, subkey = jax.random.split(key)
            results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

            key, subkey = jax.random.split(key)
            mse_vs_point_ratio = get_mse_vs_point_ratio(
                key=subkey,
                autoencoder=trainer.autoencoder,
                state=results["state"],
                enc_point_ratio_test_list=enc_point_ratio_test_list,
                dataloader=test_dataloader,
            )

            save_model_results(
                autoencoder=trainer.autoencoder,
                results=results,
                model_dir=os.path.join(
                    output_dir, "models", str(run_idx), str(enc_point_ratio_train)
                ),
            )

            save_data_results(
                autoencoder=trainer.autoencoder,
                results=results,
                test_dataloader=test_dataloader,
                data_dir=os.path.join(
                    output_dir, "data", str(run_idx), str(enc_point_ratio_train)
                ),
                additional_data={
                    "mse_vs_point_ratio": mse_vs_point_ratio,
                    "train_point_ratio": enc_point_ratio_train,
                },
            )


def get_mse_vs_point_ratio(
    key, autoencoder, state, enc_point_ratio_test_list, dataloader
):
    mse_vs_point_ratio = {}
    for enc_point_ratio_test in tqdm(enc_point_ratio_test_list):
        sum_total_mse = 0
        for u, x, _, _ in dataloader:
            n_total_pts = u.shape[1]
            n_rand_pts = int(enc_point_ratio_test * n_total_pts)

            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey, n_total_pts, (n_rand_pts,), replace=False
            )

            u_partial = u[:, indices, :]
            x_partial = x[:, indices, :]

            vars = {"params": state.params, "batch_stats": state.batch_stats}
            u_hat = autoencoder.apply(vars, u_partial, x_partial, x)

            sum_batch_mse = jnp.sum(jnp.mean(jnp.sum((u - u_hat) ** 2, axis=2), axis=1))
            sum_total_mse += sum_batch_mse

        mse_vs_point_ratio[enc_point_ratio_test] = sum_total_mse / len(
            dataloader.dataset
        )
    return mse_vs_point_ratio
