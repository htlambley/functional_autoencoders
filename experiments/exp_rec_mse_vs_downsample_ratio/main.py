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
    get_raw_x,
    yaml_load,
    fit_trainer_using_config,
)


def run_rec_mse_vs_downsample_ratio(
    key,
    output_dir,
    config_path,
    n_runs,
    ns_viscosity,
    downsample_ratios,
    enc_point_ratio_train,
    verbose="metrics",
):

    config = yaml_load(config_path)

    mask_train = ComplementMasking(enc_point_ratio_train)
    train_dataloader, test_dataloader = get_dataloaders(
        NavierStokes, data_base=".", viscosity=ns_viscosity, transform_train=mask_train
    )

    for run_idx in range(n_runs):
        key, subkey = jax.random.split(key)
        trainer = get_trainer(subkey, config, train_dataloader, test_dataloader)

        key, subkey = jax.random.split(key)
        results = fit_trainer_using_config(subkey, trainer, config, verbose=verbose)

        mse_vs_size = get_mse_vs_size(
            autoencoder=trainer.autoencoder,
            state=results["state"],
            downsample_ratios=downsample_ratios,
            dataloader=test_dataloader,
        )

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
            additional_data={"mse_vs_size": mse_vs_size},
        )


def get_mse_vs_size(autoencoder, state, downsample_ratios, dataloader):
    mse_vs_size = {}
    for ratio in tqdm(downsample_ratios):
        total_mse = 0
        for u, x, _, _ in dataloader:
            n_batch = u.shape[0]
            n = int(u.shape[1] ** 0.5)
            u_down = u.reshape(n_batch, n, n)[:, ::ratio, ::ratio]
            n_down = u_down.shape[1]

            x_down_single_batch = get_raw_x(n_down, n_down).reshape(-1, 2)
            x_down = jnp.repeat(x_down_single_batch[None, ...], n_batch, axis=0)
            u_down = u_down.reshape(n_batch, -1, 1)

            vars = {"params": state.params, "batch_stats": state.batch_stats}
            u_hat = autoencoder.apply(vars, u_down, x_down, x)

            sum_batch_mse = jnp.sum(jnp.mean(jnp.sum((u - u_hat) ** 2, axis=2), axis=1))
            total_mse += sum_batch_mse / len(dataloader)

        mse_vs_size[n_down] = total_mse / len(dataloader)
    return mse_vs_size
