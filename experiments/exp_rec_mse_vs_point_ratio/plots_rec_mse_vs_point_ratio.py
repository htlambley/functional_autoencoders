import os
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plots import get_cmap
from functional_autoencoders.util import get_raw_x, pickle_load


def plot_error_distribution_per_index(
        key,    
        n_runs, 
        u_full_ref,
        state_high, 
        state_low, 
        autoencoder_high, 
        autoencoder_low,
        enc_point_ratio_test_high,
        enc_point_ratio_test_low,
        test_dataloader_full, 
        save_dir,
    ):

    key, subkey = jax.random.split(key)
    error_list_high, x_enc_list_high = get_error_and_x_enc_lists(
        key=subkey, 
        state=state_high, 
        autoencoder=autoencoder_high, 
        u_full=u_full_ref, 
        enc_point_ratio_test=enc_point_ratio_test_high, 
        n_runs=n_runs, 
    )

    _, x_enc_list_sorted_high = zip(*sorted(zip(error_list_high, x_enc_list_high), key=lambda x: x[0]))
    key, subkey = jax.random.split(key)
    error_list_low, x_enc_list_low = get_error_and_x_enc_lists(
        key=subkey, 
        state=state_low, 
        autoencoder=autoencoder_low, 
        u_full=u_full_ref, 
        enc_point_ratio_test=enc_point_ratio_test_low, 
        n_runs=n_runs,
    )
    _, x_enc_list_sorted_low = zip(*sorted(zip(error_list_low, x_enc_list_low), key=lambda x: x[0]))

    indices_k_nearest_neighbors = get_k_nearest_neighbors_indices(u_full_ref, test_dataloader_full, 5)
    indices_random_samples = get_random_sample_indices(subkey, test_dataloader_full, 5)

    for j, indices in enumerate([indices_k_nearest_neighbors, indices_random_samples]):
        fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        gs = gridspec.GridSpec(2, len(indices), figure=fig, height_ratios=[1, 1])

        for i, idx in enumerate(indices.tolist()):
            u_full, x_full, _, _ = test_dataloader_full.dataset[idx]
            u_full = jnp.array(u_full)
            x_full = jnp.array(x_full)

            ax1 = fig.add_subplot(gs[0, i])
            ax1.imshow(u_full.reshape(64, 64), cmap='hot')
            ax1.set_title(f'idx={idx}')
            ax1.axis('off')

            single_error_low = get_single_error_for_x_enc(autoencoder_low, state_low, u_full, x_enc_list_sorted_low[0])
            single_error_high = get_single_error_for_x_enc(autoencoder_high, state_high, u_full, x_enc_list_sorted_high[0])

            ax2 = fig.add_subplot(gs[1, i])

            key, subkey = jax.random.split(key)
            plot_error_distribution(subkey, state_high, autoencoder_high, test_dataloader_full, 
                                    idx, n_runs, label='High', color='k', ax=ax2)
            ax2.axvline(x=single_error_high.item(), color='k', linestyle='--')

            key, subkey = jax.random.split(key)
            plot_error_distribution(subkey, state_low, autoencoder_low, test_dataloader_full, 
                                    idx, n_runs, label='Low', color='r', ax=ax2)
            ax2.axvline(x=single_error_low.item(), color='r', linestyle='--')

            if i == len(indices) - 1:
                ax2.legend()

            if i != 0:
                ax2.set_ylabel('')

        if save_dir is not None:
            plt.savefig(f'{save_dir}/good_config_with_knn.pdf' if j == 0 else f'{save_dir}/good_config_with_random_samples.pdf')
        plt.show()
        plt.close()


def plot_error_vs_point_ratio(
        key, 
        idx,
        n_runs, 
        state_high, 
        state_low, 
        autoencoder_high, 
        autoencoder_low, 
        test_dataloader_full, 
        enc_point_ratio_test_list,
        save_dir,
    ):

    key, subkey = jax.random.split(key)
    errors_high_per_ratio_rand_pts_enc_test = get_errors_per_ratio(
        key=subkey, 
        idx=idx, 
        n_runs=n_runs, 
        state=state_high, 
        autoencoder=autoencoder_high, 
        test_dataloader_full=test_dataloader_full, 
        enc_point_ratio_test_list=enc_point_ratio_test_list,
    )

    key, subkey = jax.random.split(key)
    errors_low_per_ratio_rand_pts_enc_test = get_errors_per_ratio(
        key=subkey, 
        idx=idx, 
        n_runs=n_runs, 
        state=state_low, 
        autoencoder=autoencoder_low, 
        test_dataloader_full=test_dataloader_full, 
        enc_point_ratio_test_list=enc_point_ratio_test_list,
    )

    max_error = max(
        [max(errors) for errors in errors_low_per_ratio_rand_pts_enc_test.values()] + 
        [max(errors) for errors in errors_high_per_ratio_rand_pts_enc_test.values()]
    )

    color_palette = sns.color_palette('Reds', len(errors_high_per_ratio_rand_pts_enc_test))

    for i, (enc_point_ratio_test, errors) in enumerate(errors_high_per_ratio_rand_pts_enc_test.items()):
        sns.kdeplot(jnp.array(errors).flatten(), label=int(enc_point_ratio_test * 100), color=color_palette[i])

    plt.xlim(0, max_error)
    plt.xlabel('MSE')
    plt.legend(title='Point \% (evaluation)')
    if save_dir is not None:
        plt.savefig(f'{save_dir}/high_model_mse.pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    for enc_point_ratio_test, errors in errors_low_per_ratio_rand_pts_enc_test.items():
        sns.kdeplot(jnp.array(errors).flatten(), color=color_palette.pop(0))

    plt.xlim(0, max_error)
    plt.xlabel('MSE')
    if save_dir is not None:
        plt.savefig(f'{save_dir}/low_model_mse.pdf', bbox_inches='tight')
    plt.show()


def plot_point_ratios(data_output_dir, save_dir=None):
    data_pt_ratios = get_mse_results(data_output_dir)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3.3, 3.3)

    cs = get_cmap("Reds", 3)
    linestyles = ["-", "--", ":"]
    shapes = ["o", "s", "^"]
    for i, train_pt_ratio in enumerate(data_pt_ratios.keys()):
        test_pt_ratios = data_pt_ratios[train_pt_ratio].keys()

        mse_means = np.array(
            [np.mean(data_pt_ratios[train_pt_ratio][r]) for r in test_pt_ratios]
        )
        mse_stds = np.array(
            [np.std(data_pt_ratios[train_pt_ratio][r]) for r in test_pt_ratios]
        )

        test_pt_ratios = [r * 100 for r in test_pt_ratios]

        ax.plot(
            test_pt_ratios,
            mse_means,
            ls=linestyles[i],
            label=f"{(train_pt_ratio * 100):.0f}%",
            color=cs[i],
        )
        ax.scatter(test_pt_ratios, mse_means, marker=shapes[i], color=cs[i])
        ax.fill_between(
            test_pt_ratios,
            mse_means - 2 * mse_stds,
            mse_means + 2 * mse_stds,
            color=cs[i],
            alpha=0.2,
        )

    ax.set_ymargin(0.08)
    ax.set_yscale("log", base=10)
    ax.set_ylim(bottom=4.5e-4, top=14.5e-4)
    ax.set_ylabel(r"MSE [$\times 10^{-4}$]")
    ax.xaxis.set_tick_params(top=False, which="both")
    ax.yaxis.set_tick_params(right=False, which="both")
    ax.legend(title="Point \\% (train)")

    if save_dir is not None:
        fig.savefig(f"{save_dir}/ns_point_ratios.pdf")


def plot_error_distribution(key, state, autoencoder, test_dataloader, idx, n_runs, **plot_kwargs):
    key, subkey = jax.random.split(key)
    error_list_low, _ = get_error_and_x_enc_lists(
        key=subkey, 
        state=state, 
        autoencoder=autoencoder, 
        u_full=jnp.array(test_dataloader.dataset[idx][0]),
        enc_point_ratio_test=0.1, 
        n_runs=n_runs, 
    )
    plt.hist(error_list_low, bins=100)
    sns.kdeplot(jnp.array(error_list_low).flatten(), **plot_kwargs)


def get_mse_results(data_output_dir):
    mse_results = {}
    for run_idx_str in os.listdir(data_output_dir):
        for train_point_ratio in os.listdir(os.path.join(data_output_dir, run_idx_str)):
            result = pickle_load(
                os.path.join(data_output_dir, run_idx_str, train_point_ratio, "data.pickle")
            )
            mse_vs_point_ratio = result["additional_data"]["mse_vs_point_ratio"]
            train_point_ratio = result["additional_data"]["train_point_ratio"]

            if train_point_ratio not in mse_results:
                mse_results[train_point_ratio] = {}

            for eval_point_ratio, mse in mse_vs_point_ratio.items():
                if eval_point_ratio not in mse_results[train_point_ratio]:
                    mse_results[train_point_ratio][eval_point_ratio] = []

                mse_results[train_point_ratio][eval_point_ratio].append(mse)

    for train_point_ratio in mse_results:
        mse_results[train_point_ratio] = {
            k: v
            for k, v in sorted(
                mse_results[train_point_ratio].items(), key=lambda item: item[0]
            )
        }

    mse_results = {k: v for k, v in sorted(mse_results.items(), key=lambda item: item[0])}
    return mse_results


def get_errors_per_ratio(key, idx, n_runs, state, autoencoder, test_dataloader_full, enc_point_ratio_test_list):
    errors_per_ratio_rand_pts_enc_test = {}
    for enc_point_ratio_test in enc_point_ratio_test_list:
        key, subkey = jax.random.split(key)
        error_list, _ = get_error_and_x_enc_lists(
            key=subkey, 
            state=state, 
            autoencoder=autoencoder, 
            u_full=jnp.array(test_dataloader_full.dataset[idx][0]),
            enc_point_ratio_test=enc_point_ratio_test, 
            n_runs=n_runs, 
        )

        errors_per_ratio_rand_pts_enc_test[enc_point_ratio_test] = error_list

    return errors_per_ratio_rand_pts_enc_test


def get_binary_mask(x_sparse, N=64):
    x_full = get_raw_x(N, N)
    u_binary = jnp.zeros((N, N)).reshape(-1, 1)

    # Set u_binary to 1 where x_full value is equal to x_values
    # x_full is n_full x 2 and x_values is n_sparse x 2
    for x_value in x_sparse:
        u_binary += jnp.prod(x_full == x_value, axis=1).reshape(-1, 1)

    return u_binary.flatten()


def squared_norm(u, x):
    return jnp.mean(jnp.sum(u**2, axis=2), axis=1)


def get_single_error_for_x_enc(autoencoder, state, u_full, x_enc_in):
    n = int(u_full.shape[0] ** 0.5)
    x_full = get_raw_x(n, n)
    u_mask = get_binary_mask(x_enc_in)
    u_enc = u_full[u_mask == 1]
    x_enc = x_full[u_mask == 1]

    vars = {'params': state.params, 'batch_stats': state.batch_stats}
    u_hat = autoencoder.apply(vars, u_enc[None], x_enc[None], x_full[None])

    single_error = jnp.mean(jnp.sum((u_full - u_hat)**2, axis=2), axis=1)
    return single_error


def get_single_error_and_x_enc(key, autoencoder, state, u_full, x_full=None, ratio_rand_pts_enc_test=0.1):
    if x_full is None:
        n = int(u_full.shape[0] ** 0.5)
        x_full = get_raw_x(n)

    n_total_pts = u_full.shape[0]
    n_rand_pts = int(ratio_rand_pts_enc_test * n_total_pts)

    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, n_total_pts, (n_rand_pts,), replace=False)

    u_partial = u_full[indices, :]
    x_partial = x_full[indices, :]

    vars = {'params': state.params, 'batch_stats': state.batch_stats}
    u_hat = autoencoder.apply(vars, u_partial[None], x_partial[None], x_full[None])

    single_error = jnp.mean(jnp.sum((u_full - u_hat)**2, axis=2), axis=1)
    return single_error, x_partial


def get_error_and_x_enc_lists(key, state, autoencoder, u_full, enc_point_ratio_test, n_runs):
    error_list = []
    x_enc_list = []

    n = int(u_full.shape[0] ** 0.5)
    x_full = get_raw_x(n, n)

    for _ in range(n_runs):
        key, subkey = jax.random.split(key)
        single_error, x_partial = get_single_error_and_x_enc(subkey, autoencoder, state, u_full, x_full, enc_point_ratio_test)
        error_list.append(single_error.item())
        x_enc_list.append(x_partial)

    return error_list, x_enc_list


def get_k_nearest_neighbors_indices(u_target, dataloader, k):
    u_target = jnp.array(u_target)
    u_list = [jnp.array(u[0]) for u in dataloader.dataset]
    distances = [jnp.sum((u_target - u_ref)**2) for u_ref in u_list]
    nearest_indices = jnp.argsort(jnp.array(distances))[:k]
    return nearest_indices


def get_random_sample_indices(key, dataloader, n_samples):
    key, subkey = jax.random.split(key)
    n_samples = min(n_samples, len(dataloader.dataset))
    return jax.random.choice(subkey, jnp.arange(len(dataloader.dataset)), shape=(n_samples,), replace=False)
