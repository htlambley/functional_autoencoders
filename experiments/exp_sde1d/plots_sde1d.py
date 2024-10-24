import sys

sys.path.append("../")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
import seaborn as sns
from functools import partial
from functional_autoencoders.samplers.sampler_vae import SamplerVAE
from functional_autoencoders.util.random.sde import add_bm_noise
from plots import get_cmap


def plot_potential_and_samples(ax1, ax2, potential_1d, train_dataloader, n_samples=5):
    x = np.linspace(-1.5, 1.62, 1000)
    U = potential_1d(x[None, :, None])
    s = 0
    shift = -np.min(U) + s
    ax1.plot(x, U + shift, c="r", zorder=3)
    ax1.set_xticks([-1, 1])
    ax1.set_xticklabels(["$x_{1} = -1$", "$x_{2} = +1$"])
    ax1.set_yticks([0, shift + potential_1d([-1])])
    ax1.set_ymargin(0)
    ax1.yaxis.set_tick_params(right=False)
    ax1.xaxis.set_tick_params(top=False)
    ax1.set_xlabel("$x$")
    ax1.set_title("(a) Potential $U(x)$")

    u, x, _, _ = next(iter(train_dataloader))
    cs = get_cmap("Reds", n_samples)
    for i in range(0, n_samples):
        ax2.plot(x[i], u[i], c=cs[i])
    ax2.set_xlabel("$t$")
    ax2.set_title(r"(b) Sample paths $(u_{t})_{t \in [0, 5]}$")
    ax2.set_yticks([-1, 0, 1])
    ax2.yaxis.set_tick_params(right=False)
    ax2.xaxis.set_tick_params(top=False)


def plot_reconstructions_and_generated_samples(
    key,
    ax1,
    ax2,
    ax3,
    info,
    theta,
    config_data,
    test_dataloader,
    n_samples=5,
    title=False,
):
    autoencoder = info["autoencoder"]
    state = info["results"]["state"]

    u, x, _, _ = next(iter(test_dataloader))

    vars = {"params": state.params, "batch_stats": state.batch_stats}
    uhat = autoencoder.apply(vars, u, x, x)

    cs_u = get_cmap("Reds", n_samples)
    # cs_uhat = get_cmap("GnBu", n_samples)
    for i in range(n_samples):
        ax1.plot(x[i], u[i], c=cs_u[i])
        ax1.plot(x[i], uhat[i], c="k", zorder=3, linewidth=2, linestyle=(0, (3, 0.5)))

    ax1.yaxis.set_tick_params(right=False)
    ax1.xaxis.set_tick_params(top=False)
    ax1.set_yticks([-1, 0, 1])
    if title:
        ax1.set_title("(a) Reconstructions")
    cs_u = get_cmap("Reds", n_samples)

    key, subkey = jax.random.split(key)
    sampler = SamplerVAE(autoencoder, state)
    samples = sampler.sample(x[:n_samples], subkey)

    s = add_bm_noise(
        samples=samples,
        epsilon=config_data["epsilon"],
        theta=theta,
        sim_dt=config_data["sim_dt"],
        T=config_data["T"],
    )

    for i in range(n_samples):
        ax2.plot(x[i], s[i], c=cs_u[i])
    ax2.yaxis.set_tick_params(right=False)
    ax2.xaxis.set_tick_params(top=False)
    ax2.set_yticks([-1, 0, 1])
    if title:
        ax2.set_title(r"(b) Realizations of $g(z; \psi) + \eta$")

    n_repeats = 1000
    for i in range(n_samples):
        s = jnp.expand_dims(samples[i, :, :], 0)
        s = jnp.repeat(s, n_repeats, axis=0)
        s = add_bm_noise(
            samples=s,
            epsilon=config_data["epsilon"],
            theta=theta,
            sim_dt=config_data["sim_dt"],
            T=config_data["T"],
        )

        std = jnp.std(s[:, :, 0], axis=0)
        if i == 0:
            ax3.plot(x[i], samples[i], c=cs_u[i], label=r"$g(z; \psi)$")
            ax3.fill_between(
                x[i, :, 0],
                samples[i, :, 0] - std,
                samples[i, :, 0] + std,
                color=cs_u[i],
                alpha=0.4,
                label="1 SD",
            )
        else:
            ax3.plot(x[i], samples[i], c=cs_u[i])
            ax3.fill_between(
                x[i, :, 0],
                samples[i, :, 0] - std,
                samples[i, :, 0] + std,
                color=cs_u[i],
                alpha=0.4,
            )

    ax3.yaxis.set_tick_params(right=False)
    ax3.xaxis.set_tick_params(top=False)
    ax3.set_yticks([-1, 0, 1])
    if title:
        ax3.set_title(r"(c) Distribution of $g(z; \psi) + \eta$")


def plot_latent_variable(
    ax,
    ax_colorbar,
    info,
    test_dataloader,
    z_min=-2.5,
    z_max=2.5,
    n_evals=8,
):
    autoencoder = info["autoencoder"]
    state = info["results"]["state"]

    z = jnp.linspace(z_min, z_max, n_evals)
    z = jnp.expand_dims(z, 1)
    _, x, _, _ = next(iter(test_dataloader))

    # samples = trainer.autoencoder.decode(variables, z, x[: z.shape[0], :, :])
    samples = autoencoder.decode(state, z, x[: z.shape[0], :, :], train=False)

    cs_u = get_cmap("Reds", z.shape[0], start=0.3)
    for i in range(z.shape[0]):
        ax.plot(x[i, :, 0], samples[i, :, 0], c=cs_u[i])
    ax.set_yticks([-1, 0, 1])
    ax.xaxis.set_tick_params(right=False)
    ax.yaxis.set_tick_params(top=False)
    ax.set_title(r"(a) $g(z; \psi)(t)$ for $z \in [-2.5, 2.5]$")

    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)),
        )
        return new_cmap

    matplotlib.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=truncate_colormap(matplotlib.cm.Reds, minval=0.3),
        orientation="horizontal",
        norm=matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5),
    )
    ax_colorbar.set_ylabel("$z$", rotation=0)
    ax_colorbar.yaxis.set_label_coords(-0.025, -0.03)


@partial(jax.vmap, in_axes=(0, 0))
def get_transition_times(path, x):
    return x[jnp.argmax(path[:, 0] > 0), 0]


def plot_transition_time_distribution(
    key,
    ax,
    info,
    test_dataloader,
):
    autoencoder = info["autoencoder"]
    state = info["results"]["state"]

    tts_samples = []
    tts_u = []
    for u, x, _, _ in iter(test_dataloader):
        key, subkey = jax.random.split(key)
        sampler = SamplerVAE(autoencoder, state)
        samples = sampler.sample(x, subkey)

        tt_samples = get_transition_times(samples, x)
        tt_u = get_transition_times(u, x)
        tts_samples = tts_samples + [*tt_samples.tolist()]
        tts_u = tts_u + [*tt_u.tolist()]

    sns.kdeplot(tts_samples, ax=ax, label="FVAE", color="k", fill=True, alpha=0.5)
    sns.kdeplot(
        tts_u,
        ax=ax,
        label="Direct numerical simulation",
        color="r",
        fill=True,
        alpha=0.5,
    )

    ax.legend()
    ax.yaxis.set_tick_params(right=False)
    ax.xaxis.set_tick_params(top=False)
    ax.set_title(f"(b) Time $t$ of first crossing above $0$ (N={len(tts_samples)})")
    ax.set_ylabel("Density")
