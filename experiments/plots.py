import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches as mpatches
from tqdm.auto import tqdm
from functional_autoencoders.util import get_n_params, get_raw_x
from functional_autoencoders.util.masks import get_mask_uniform
from functional_autoencoders.samplers.sampler_vae import SamplerVAE


def get_cmap(scheme, steps, start=0.5, stop=1):
    return matplotlib.colormaps[scheme](np.linspace(start, stop, steps))


def set_plot_style_publication():
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Computer Modern"]
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["axes.xmargin"] = 0
    matplotlib.rcParams["axes.ymargin"] = 0.02
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["lines.linewidth"] = 1.0
    matplotlib.rcParams["lines.markersize"] = 3.0


def plot_samples_2d_grid(
    state,
    autoencoder,
    x,
    plots_h,
    plots_w,
    key,
    title="Samples",
):
    key, subkey = jax.random.split(key)
    sampler = SamplerVAE(autoencoder, state)
    samples = sampler.sample(x, subkey)

    grid_pts = int(x.shape[1] ** 0.5)

    for i in range(x.shape[0]):
        u = samples[i]
        plt.subplot(plots_h, plots_w, i + 1)
        plt.imshow(u.reshape(grid_pts, grid_pts), cmap="hot")
        plt.axis("off")
        if i == 0:
            plt.title(title)


def plot_train_val_losses(losses_train, losses_val, start_idx_train=0):
    n_plots = 1 + len(losses_val)

    plt.subplot(1, n_plots, 1)
    plt.plot(losses_train[start_idx_train:])
    plt.title("Train Loss")

    for i, metric_name in enumerate(losses_val):
        plt.subplot(1, n_plots, i + 2)
        plt.plot(losses_val[metric_name])
        plt.yscale("log")
        plt.title(metric_name)


def plot_dataset(batch, n_samples, title="Dataset", is_step=False):
    plot_fn = plt.step if is_step else plt.plot
    u, x = batch
    for i in range(n_samples):
        plot_fn(x[i], u[i])
    plt.title(title)


def plot_reconstructions_1d(
    variables,
    autoencoder,
    init_batch,
    n_samples,
    set_legend=True,
    title="Reconstructions",
    is_step=False,
):
    u, x = init_batch
    plot_fn = plt.step if is_step else plt.plot
    u_hat = autoencoder.apply(variables, u[:n_samples], x[:n_samples], x[:n_samples])

    for i in range(n_samples):
        plot_fn(x[i], u[i], color="b")
    for i in range(n_samples):
        plot_fn(x[i], u_hat[i], color="r")

    plt.title(title)

    if set_legend:
        plt.legend(
            handles=[
                mpatches.Patch(color="b", label="True"),
                mpatches.Patch(color="r", label="Reconstructed"),
            ]
        )


def plot_dataset_samples(samples, subplots_per_row=8):
    n = int(math.sqrt(samples.shape[1]))
    subplot_height = math.ceil(samples.shape[0] / subplots_per_row)
    for i in range(samples.shape[0]):
        u = samples[i]
        plt.subplot(subplot_height, subplots_per_row, i + 1)
        plt.imshow(u.reshape(n, n), cmap="hot", vmin=0, vmax=1)
        plt.axis("off")


def plot_training_results(results):
    plot_train_val_losses(
        results["training_loss_history"], results["metrics_history"], start_idx_train=3
    )
    plt.tight_layout()
    plt.show()

    n_params = get_n_params(results["state"].params)
    metric_names = results["metrics_history"].keys()
    for metric_name in reversed(metric_names):
        print(f'{metric_name}: {results["metrics_history"][metric_name][-1]:.3e}')
    print(f"Number of parameters: {n_params}")


def plot_reconstructions(autoencoder, state, dataloader_iter, n_recs):
    u, x, _, _ = next(dataloader_iter)
    u_hat = autoencoder.apply(
        {"params": state.params, "batch_stats": state.batch_stats}, u, x, x
    )
    n = int(u.shape[1] ** 0.5)

    for i in range(n_recs):
        vmin = min(jnp.min(u[i]), jnp.min(u_hat[i]))
        vmax = max(jnp.max(u[i]), jnp.max(u_hat[i]))

        plt.subplot(3, n_recs, i + 1)
        plt.imshow(u[i].reshape(n, n), cmap="hot", vmin=vmin, vmax=vmax)
        plt.title(f"Original ({i})")
        plt.axis("off")

        plt.subplot(3, n_recs, i + 1 + n_recs)
        plt.imshow(u_hat[i].reshape(n, n), cmap="hot", vmin=vmin, vmax=vmax)
        plt.title(f"Rec ({i})")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(3, n_recs, i + 1 + 2 * n_recs)
        plt.imshow(jnp.square(u[i] - u_hat[i]).reshape(n, n), cmap="hot")
        plt.title(f"Error ({i})")
        plt.colorbar()
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_different_resolutions(key, autoencoder, state, pts_list, n_samples):
    for pts in pts_list:
        x = get_raw_x(pts, pts)
        x = jnp.tile(jnp.expand_dims(x, 0), (n_samples, 1, 1))

        key, subkey = jax.random.split(key)
        plot_samples_2d_grid(
            state=state,
            autoencoder=autoencoder,
            x=x,
            plots_h=1,
            plots_w=n_samples,
            key=subkey,
            title=f"Resolution: {pts}x{pts}",
        )
        plt.tight_layout()
        plt.show()


def plot_masked_reconstructions(
    key,
    autoencoder,
    state,
    dataloader_iter,
    get_mask_fn,
    n_recs=1,
    downsample_ratio=1,
    cmap="hot",
    save_dir=None,
):
    N_ROW_PLOTS = 1
    N_COL_PLOTS = 4

    u, x, _, _ = next(iter(dataloader_iter))

    for i in range(n_recs):
        u_orig = u[i : i + 1]
        x_orig = x[i : i + 1]

        vmin = jnp.min(u_orig)
        vmax = jnp.max(u_orig)

        n = int(u_orig.shape[1] ** 0.5)

        u_orig_down = u_orig.reshape(n, n)[::downsample_ratio, ::downsample_ratio]
        n_down = u_orig_down.shape[0]
        x_orig_down = get_raw_x(n_down, n_down).reshape(1, -1, 2)
        u_orig_down = u_orig_down.reshape(1, -1, 1)

        key, subkey = jax.random.split(key)
        mask = get_mask_fn(subkey, u_orig_down)

        u_mask_partial = u_orig_down[:, ~mask, :]
        x_mask_partial = x_orig_down[:, ~mask, :]
        u_mask = u_orig_down.copy()
        u_mask[:, mask] = 0

        vars = {"params": state.params, "batch_stats": state.batch_stats}
        u_rec = autoencoder.apply(vars, u_mask_partial, x_mask_partial, x_orig)

        u_orig = u_orig.reshape(n, n)
        u_rec = u_rec.reshape(n, n)
        u_mask = u_mask.reshape(n_down, n_down)

        if save_dir is None:
            plt.subplot(N_ROW_PLOTS, N_COL_PLOTS, 1)
            plt.title("Original")
        plt.imshow(u_orig, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if save_dir is not None:
            plt.savefig(f"{save_dir}/original_{i}.pdf")
            plt.close()

        if save_dir is None:
            plt.subplot(N_ROW_PLOTS, N_COL_PLOTS, 2)
            plt.title("Masked")
        plt.imshow(u_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if save_dir is not None:
            plt.savefig(f"{save_dir}/masked_{i}.pdf")
            plt.close()

        if save_dir is None:
            plt.subplot(N_ROW_PLOTS, N_COL_PLOTS, 3)
            plt.title("Reconstructed")
        plt.imshow(u_rec, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        if save_dir is not None:
            plt.colorbar()
            plt.savefig(f"{save_dir}/rec_{i}.pdf")
            plt.close()

        if save_dir is None:
            plt.subplot(N_ROW_PLOTS, N_COL_PLOTS, 4)
            plt.title("|Original - Mean|")
        plt.imshow(jnp.abs(u_orig - u_rec), cmap=cmap)
        plt.axis("off")
        if save_dir is not None:
            plt.colorbar()
            plt.savefig(f"{save_dir}/diff_{i}.pdf")
            plt.close()

        if save_dir is None:
            plt.tight_layout()
            plt.show()
            plt.close()


def plot_uniform_reconstruction_range(
    key,
    autoencoder,
    state,
    dataloader_iter,
    mask_ratios,
    n_recs=1,
    cmap="hot",
    save_dir=None,
):

    u, x, _, _ = next(iter(dataloader_iter))

    for i in range(n_recs):
        u_orig = u[i : i + 1]
        x_orig = x[i : i + 1]

        vmin = jnp.min(u_orig)
        vmax = jnp.max(u_orig)

        n = int(u_orig.shape[1] ** 0.5)

        for j, mask_ratio in enumerate(mask_ratios):
            get_mask_fn = lambda key, u: get_mask_uniform(key, u, mask_ratio=mask_ratio)

            if mask_ratio > 0:
                key, subkey = jax.random.split(key)
                mask = get_mask_fn(subkey, u_orig)

                u_mask_partial = u_orig[:, ~mask, :]
                x_mask_partial = x_orig[:, ~mask, :]
                u_mask = u_orig.copy()
                u_mask[:, mask] = 0
            else:
                u_mask_partial = u_orig
                x_mask_partial = x_orig
                u_mask = u_orig

            vars = {"params": state.params, "batch_stats": state.batch_stats}
            u_rec = autoencoder.apply(vars, u_mask_partial, x_mask_partial, x_orig)

            u_rec = u_rec.reshape(n, n)
            u_mask = u_mask.reshape(n, n)

            if save_dir is None:
                plt.subplot(2, len(mask_ratios), j + 1)
                if j == 0:
                    plt.ylabel("Original + Mask")
                plt.title(f"{mask_ratio}")
            plt.imshow(u_mask, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.yticks([])
            plt.xticks([])
            if save_dir is not None:
                plt.savefig(f"{save_dir}/original_mask_{mask_ratio}_{i}.pdf")
                plt.close()

            if save_dir is None:
                plt.subplot(2, len(mask_ratios), j + 1 + len(mask_ratios))
                if j == 0:
                    plt.ylabel("Reconstructed")
            plt.imshow(u_rec, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.yticks([])
            plt.xticks([])
            if save_dir is not None:
                plt.colorbar()
                plt.savefig(f"{save_dir}/reconstructed_{mask_ratio}_{i}.pdf")
                plt.close()

        if save_dir is None:
            plt.tight_layout()
            plt.show()


def plot_upsamples(
    autoencoder,
    state,
    dataloader_iter,
    downsample_ratios,
    n_recs=1,
    cmap="hot",
    save_dir=None,
):

    u, x, _, _ = next(iter(dataloader_iter))

    for i in range(n_recs):
        u_orig = u[i : i + 1]
        x_orig = x[i : i + 1]

        vmin = jnp.min(u_orig)
        vmax = jnp.max(u_orig)

        n = int(u_orig.shape[1] ** 0.5)

        for j, ratio in enumerate(downsample_ratios):
            u_orig_down = u_orig.reshape(n, n)[::ratio, ::ratio]
            n_down = u_orig_down.shape[0]

            x_orig_down = get_raw_x(n_down, n_down).reshape(1, -1, 2)
            u_orig_down = u_orig_down.reshape(1, -1, 1)

            vars = {"params": state.params, "batch_stats": state.batch_stats}
            u_rec = autoencoder.apply(vars, u_orig_down, x_orig_down, x_orig)

            u_orig_down = u_orig_down.reshape(n_down, n_down)
            u_rec = u_rec.reshape(n, n)

            if save_dir is None:
                plt.subplot(2, len(downsample_ratios), j + 1)
                if j == 0:
                    plt.ylabel("Original")
                plt.title(f"{ratio}")
            plt.imshow(u_orig_down, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.yticks([])
            plt.xticks([])
            if save_dir is not None:
                plt.savefig(f"{save_dir}/original_{ratio}_{i}.pdf")
                plt.close()

            if save_dir is None:
                plt.subplot(2, len(downsample_ratios), j + 1 + len(downsample_ratios))
                if j == 0:
                    plt.ylabel("Upsampled Reconstruction")
            plt.imshow(u_rec, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.yticks([])
            plt.xticks([])
            if save_dir is not None:
                plt.colorbar()
                plt.savefig(f"{save_dir}/upsampled_mean_{ratio}_{i}.pdf")
                plt.close()

        if save_dir is None:
            plt.tight_layout()
            plt.show()


def plot_nearest_neighbours(u, dataset, k):
    n = int(u.shape[0] ** 0.5)
    u = u.reshape(n, n)

    u_dataset = [jnp.array(u_ref[0].reshape(n, n)) for u_ref in dataset]
    distances = [jnp.sum((u - u_ref) ** 2) for u_ref in u_dataset]
    nearest_indices = jnp.argsort(jnp.array(distances))[:k]

    plt.subplot(1, k + 1, 1)
    plt.imshow(u, cmap="hot", vmin=0, vmax=1)
    plt.title("Query")
    plt.axis("off")

    for i in range(k):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(u_dataset[nearest_indices[i]], cmap="hot", vmin=0, vmax=1)
        plt.title(f"Neighbour {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_latent_interpolation(
    autoencoder, state, alpha_list, u_start, u_end, z_start, z_end, save_dir=None
):
    n = int(u_start.shape[0] ** 0.5)
    x = get_raw_x(n, n)

    vmin = min(jnp.min(u_start), jnp.min(u_end))
    vmax = max(jnp.max(u_start), jnp.max(u_end))

    if save_dir is None:
        plt.subplot(1, len(alpha_list) + 2, 1)
        plt.title("Start True")
    plt.imshow(u_start.reshape(n, n), cmap="hot", vmin=vmin, vmax=vmax)
    plt.axis("off")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/start_true.pdf")
        plt.close()

    for i, alpha in enumerate(alpha_list, start=2):
        z_curr = (1 - alpha) * z_start + alpha * z_end
        u_rec = autoencoder.decode(state, z_curr[None, :], x[None, :, :])

        if save_dir is None:
            plt.subplot(1, len(alpha_list) + 2, i)
            t = r"$\alpha$"
            plt.title(f"Sample {t}={alpha}")
        plt.imshow(u_rec[0].reshape(n, n), cmap="hot", vmin=vmin, vmax=vmax)
        plt.axis("off")
        if save_dir is not None:
            plt.savefig(f"{save_dir}/sample_{alpha}.pdf")
            plt.close()

    if save_dir is None:
        plt.subplot(1, len(alpha_list) + 2, len(alpha_list) + 2)
        plt.title("End True")
    plt.imshow(u_end.reshape(n, n), cmap="hot", vmin=vmin, vmax=vmax)
    plt.axis("off")
    if save_dir is not None:
        plt.colorbar()
        plt.savefig(f"{save_dir}/end_true.pdf")
        plt.close()

    if save_dir is None:
        plt.tight_layout()
        plt.show()


def plot_mse_vs_mask_ratio(
    key, autoencoder, state, mask_ratios, dataloader, save_dir=None
):
    mse_per_mask_ratio = []
    for mask_ratio in tqdm(mask_ratios):
        total_mse = 0
        for u, x, _, _ in dataloader:
            key, subkey = jax.random.split(key)
            mask = get_mask_uniform(subkey, u, mask_ratio=mask_ratio)
            u_mask_partial = u[:, ~mask, :]
            x_mask_partial = x[:, ~mask, :]

            vars = {"params": state.params, "batch_stats": state.batch_stats}
            u_rec = autoencoder.apply(vars, u_mask_partial, x_mask_partial, x)

            sum_batch_mse = jnp.sum(jnp.mean(jnp.sum((u - u_rec) ** 2, axis=2), axis=1))
            total_mse += sum_batch_mse / len(dataloader)

        mse_per_mask_ratio.append(total_mse / len(dataloader))

    if save_dir is None:
        plt.title("MSE vs Mask Ratio")
        plt.xlabel("Mask Ratio")
        plt.ylabel("MSE")
    plt.plot(mask_ratios, mse_per_mask_ratio)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/mse_vs_mask_ratio.pdf")

    if save_dir is None:
        plt.show()


def plot_runs_and_medians(
    ax,
    ys,
    title,
    logx=None,
    logy=None,
    xticks=None,
    xticklabels=None,
    bottom=None,
    labelx="",
    abs=False,
):
    ax.set_xlabel(labelx)
    xs = list(ys.keys())

    if abs:
        ys = [np.abs(ys[x]) for x in xs]
    else:
        ys = [ys[x] for x in xs]

    xaug = [[i] * len(j) for (i, j) in zip(xs, ys)]
    yflat = [item for row in ys for item in row]
    xaug = [item for row in xaug for item in row]
    ax.scatter(xaug, yflat, c="k", s=5, alpha=0.5)
    medians = [np.median(arr) for arr in ys]
    ax.plot(xs, medians, "r-")

    if logx is not None:
        ax.set_xscale("log", base=logx)
        if logx == 10:
            locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
            ax.xaxis.set_major_locator(locmaj)
            locmin = matplotlib.ticker.LogLocator(
                base=10.0,
                subs=(
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                ),
                numticks=12,
            )
            ax.xaxis.set_minor_locator(locmin)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    if logy is not None:
        if logy == 10:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            formatter.set_useOffset(False)  # Disable the offset
            ax.yaxis.set_major_formatter(formatter)

    ax.set_xticks(xs if xticks is None else xticks)
    ax.xaxis.set_tick_params(top=False, which="both")
    ax.yaxis.set_tick_params(right=False, which="both")
    ax.set_xmargin(0.08)
    ax.set_ymargin(0.08)

    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([str(i) for i in xs])

    ax.set_title(title)

    if bottom is not None:
        ax.set_ylim(bottom=bottom)


def plot_wallclock(ax, losses, right_lim=100, label_fn=lambda x: x):
    cs = get_cmap("Reds", len(losses))
    linestyles = ["-", "--", ":"]
    shapes = ["o", "s", "^"]
    for i, quantity in enumerate(sorted(losses.keys())):
        xs = losses[quantity].keys()
        ys = [losses[quantity][x] for x in xs]
        ax.plot(xs, ys, ls=linestyles[i], label=label_fn(quantity), color=cs[i])
        ax.scatter(xs, ys, marker=shapes[i], color=cs[i])
    ax.set_xlim(left=0, right=right_lim)
    ax.set_ymargin(0.08)
    ax.xaxis.set_tick_params(top=False, which="both")
    ax.yaxis.set_tick_params(right=False, which="both")
