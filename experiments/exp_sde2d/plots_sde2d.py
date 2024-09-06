import sys

sys.path.append("../")

import warnings
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from functional_autoencoders.util import get_n_params
from plots import plot_train_val_losses


def get_X_Y_Z(potential, start=-0.5, end=0.5, n=100):
    X = np.linspace(start, end, n)
    Y = np.linspace(start, end, n)
    X, Y = np.meshgrid(X, Y)
    Z = potential(jnp.concat([X[..., None], Y[..., None]], axis=-1))
    return X, Y, Z


def plot_colored_line(x, y, **lc_kwargs):
    """
    Adapted from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    c = np.linspace(0, 1, x.size)  # color values for each line segment
    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    plt.gca().add_collection(lc)


def plot_contour_with_partitions(potential, x_locs, y_locs, cmap_contour="hot"):
    U, Y, Z = get_X_Y_Z(potential)
    plt.contourf(U, Y, Z, cmap=cmap_contour)

    for i in range(1, len(x_locs) - 1):
        plt.axvline(x=x_locs[i], color="black")  # Vertical lines
        plt.axhline(y=y_locs[i], color="black")  # Horizontal lines

    plt.colorbar()
    plt.xticks([])
    plt.yticks([])


def plot_contour_with_samples(
    u, potential, h, w, cmap_contour="hot", cmap_line="winter"
):

    U, Y, Z = get_X_Y_Z(potential)
    for i in range(h * w):
        plt.subplot(h, w, i + 1)
        plt.contourf(U, Y, Z, cmap=cmap_contour)
        plot_colored_line(u[i, :, 0], u[i, :, 1], cmap=cmap_line)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()


def plot_contour_with_reconstructions(
    u, u_rec, potential, cmap_contour="hot", cmap_line="winter"
):
    n_recs = u.shape[0]
    U, Y, Z = get_X_Y_Z(potential)
    for i in range(n_recs):
        plt.subplot(2, n_recs, i + 1)
        plt.contourf(U, Y, Z, cmap=cmap_contour)
        plot_colored_line(u[i, :, 0], u[i, :, 1], cmap=cmap_line)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel("True")

        plt.subplot(2, n_recs, n_recs + i + 1)
        plt.contourf(U, Y, Z, cmap=cmap_contour)
        plot_colored_line(u_rec[i, :, 0], u_rec[i, :, 1], cmap=cmap_line)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel("Reconstructed")

    plt.tight_layout()


def plot_training_results(results):
    plot_train_val_losses(
        results["training_loss_history"],
        results["metrics_history"],
        start_idx_train=3,
    )
    plt.tight_layout()
    plt.show()

    n_params = get_n_params(results["state"].params)
    metric_names = results["metrics_history"].keys()
    for metric_name in reversed(metric_names):
        print(f'{metric_name}: {results["metrics_history"][metric_name][-1]:.3e}')
    print(f"Number of parameters: {n_params}")


def plot_contour_and_heatmap(potential, cmap_contour="hot"):
    X, Y, Z = get_X_Y_Z(potential)

    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap=cmap_contour)
    ax1.set_title("Surface of Potential")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")

    ax2 = fig.add_subplot(122)
    CS = ax2.contourf(X, Y, Z, cmap=cmap_contour)
    cbar = fig.colorbar(CS, ax=ax2)
    cbar.set_label("Potential")
    ax2.set_title("Heatmap of Potential")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
