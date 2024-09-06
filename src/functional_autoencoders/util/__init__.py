"""
Utility functions.
"""

import os
import jax
import jax.numpy as jnp
import yaml
import dill as pickle
from functools import partial


def get_n_params(variables):
    """
    Computes the number of trainable parameters for the specified `variables` object.
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(variables))


def get_raw_x(h, w):
    x_mesh_list = jnp.meshgrid(
        jnp.linspace(0, 1, h + 2)[1:-1],
        jnp.linspace(0, 1, w + 2)[1:-1],
        indexing="ij",
    )
    xx = jnp.concatenate([jnp.expand_dims(v, -1) for v in x_mesh_list], axis=-1)
    xx = xx.reshape(-1, 2)
    return xx


def pickle_save(obj, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file:
        pickle.dump(obj, file)


def pickle_load(save_path):
    with open(save_path, "rb") as file:
        return pickle.load(file)


def yaml_load(save_path):
    with open(save_path, "r") as file:
        return yaml.safe_load(file)


def fit_trainer_using_config(key, trainer, config, verbose=False):
    results = trainer.fit(
        key=key,
        lr=config["trainer"]["lr"],
        lr_decay_step=config["trainer"]["lr_decay_step"],
        lr_decay_factor=config["trainer"]["lr_decay_factor"],
        max_step=config["trainer"]["max_step"],
        eval_interval=config["trainer"]["eval_interval"],
        verbose=verbose,
    )
    return results


def save_data_results(
    autoencoder, results, test_dataloader, data_dir, additional_data={}
):

    state = results["state"]
    u, x, _, _ = next(iter(test_dataloader))
    u_hat = autoencoder.apply(
        {"params": state.params, "batch_stats": state.batch_stats}, u, x, x
    )
    reconstructions = {"u": u, "u_hat": u_hat, "x": x}

    light_results = {
        "training_results": results,
        "reconstructions": reconstructions,
        "additional_data": additional_data,
    }

    path_results = os.path.join(data_dir, "data.pickle")
    pickle_save(light_results, path_results)


def save_model_results(autoencoder, results, model_dir):
    pickle_save(
        {
            "autoencoder": autoencoder,
            "results": results,
        },
        os.path.join(model_dir, "model.pkl"),
    )  # Use .pkl to ignore in git


@partial(jax.vmap, in_axes=(0, None))
def get_transition_matrix(u_bucket, n):
    P = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P = P.at[i, j].set(jnp.sum((u_bucket[1:] == j) & (u_bucket[:-1] == i)))
    row_sums = jnp.sum(P, axis=1)
    P = jnp.where(row_sums[:, None] == 0, jnp.ones_like(P) / n, P / row_sums[:, None])
    return P


@partial(jax.vmap, in_axes=(0, None, None))
def bucket_data(u, x_locs, y_locs):
    u_bucket = -jnp.ones(u.shape[0])
    for i in range(len(y_locs) - 1):
        for j in range(len(x_locs) - 1):
            mask = (
                (u[:, 0] > x_locs[j])
                & (u[:, 0] <= x_locs[j + 1])
                & (u[:, 1] > y_locs[i])
                & (u[:, 1] <= y_locs[i + 1])
            )
            u_bucket = jnp.where(mask, i * (len(x_locs) - 1) + j, u_bucket)
    return u_bucket
