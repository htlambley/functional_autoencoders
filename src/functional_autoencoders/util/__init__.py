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


def _get_transition_matrix_single(u_bucket, n):
    """Computes transition matrix from bucket sequence using vectorized one-hot encoding.

    This is more efficient than nested loops as it avoids JAX tracing overhead
    for each loop iteration and allows the computation to be fully vectorized.

    Note: Negative bucket indices (e.g., -1 for out-of-bounds points) are handled
    gracefully by jax.nn.one_hot, which returns all-zeros vectors for negative indices.
    This means transitions involving invalid buckets are simply not counted.
    """
    # Create pairs of (from_bucket, to_bucket)
    from_buckets = u_bucket[:-1].astype(jnp.int32)
    to_buckets = u_bucket[1:].astype(jnp.int32)

    # Use one-hot encoding to count transitions
    # Shape: [T-1, n] for from_buckets one-hot
    # Note: jax.nn.one_hot returns zeros for negative indices, so invalid
    # bucket indices are automatically excluded from transition counting
    from_one_hot = jax.nn.one_hot(from_buckets, n)
    # Shape: [T-1, n] for to_buckets one-hot
    to_one_hot = jax.nn.one_hot(to_buckets, n)

    # Outer product sum gives transition counts: P[i, j] = count of i -> j
    # from_one_hot.T @ to_one_hot gives [n, n] matrix
    P = jnp.einsum("ti,tj->ij", from_one_hot, to_one_hot)

    # Normalize rows
    row_sums = jnp.sum(P, axis=1)
    P = jnp.where(row_sums[:, None] == 0, jnp.ones_like(P) / n, P / row_sums[:, None])
    return P


@partial(jax.vmap, in_axes=(0, None))
def get_transition_matrix(u_bucket, n):
    return _get_transition_matrix_single(u_bucket, n)


def _bucket_data_single(u, x_locs, y_locs):
    """Assigns data points to buckets using vectorized operations.

    This is more efficient than nested loops as it uses broadcasting to
    determine bucket membership in a single vectorized pass.
    """
    n_x_bins = len(x_locs) - 1
    n_y_bins = len(y_locs) - 1

    # Use searchsorted to find which bin each coordinate falls into
    # searchsorted returns the index where the value would be inserted
    # We subtract 1 because bins are (x_locs[j], x_locs[j+1]]
    x_bin_idx = jnp.searchsorted(x_locs, u[:, 0], side="right") - 1
    y_bin_idx = jnp.searchsorted(y_locs, u[:, 1], side="right") - 1

    # Compute bucket index: i * n_x_bins + j where i is y_bin and j is x_bin
    bucket_idx = y_bin_idx * n_x_bins + x_bin_idx

    # Mark points outside the valid range as -1
    valid_x = (x_bin_idx >= 0) & (x_bin_idx < n_x_bins)
    valid_y = (y_bin_idx >= 0) & (y_bin_idx < n_y_bins)
    valid = valid_x & valid_y

    u_bucket = jnp.where(valid, bucket_idx, -1)
    return u_bucket.astype(jnp.float32)


@partial(jax.vmap, in_axes=(0, None, None))
def bucket_data(u, x_locs, y_locs):
    return _bucket_data_single(u, x_locs, y_locs)
