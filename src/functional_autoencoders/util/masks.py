import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
from functional_autoencoders.util import get_raw_x
from functional_autoencoders.util.random.grf import torus_grf


def get_mask_grf_torus(key, u, threshold, tau=3, d=2):
    key, subkey = jax.random.split(key)
    grid_pts = int(u.shape[1] ** 0.5)
    mask = (
        torus_grf(subkey, n=1, shape=(grid_pts, grid_pts), tau=tau, d=d).real.flatten()
        > threshold
    )
    return mask


def get_mask_uniform(key, u, mask_ratio):
    key, subkey = jax.random.split(key)
    mask = jax.random.bernoulli(subkey, mask_ratio, shape=(u.shape[1],))
    return mask


def get_mask_random_circle(key, u, radius):
    key, subkey = jax.random.split(key)
    random_mean = jax.random.uniform(subkey, shape=(2,))
    random_mean = random_mean * (1 - 2 * radius) + radius
    grid_pts = int(u.shape[1] ** 0.5)
    xx = get_raw_x(grid_pts, grid_pts)
    mask = jnp.linalg.norm(xx - jnp.array(random_mean), axis=1) < radius
    return mask


def get_mask_rect(key, u, h, w):
    grid_pts = int(u.shape[1] ** 0.5)

    key, k1, k2 = jax.random.split(key, 3)
    a = jax.random.randint(k1, minval=0, maxval=grid_pts - h + 1, shape=())
    b = jax.random.randint(k2, minval=0, maxval=grid_pts - w + 1, shape=())

    mask = jnp.zeros((grid_pts, grid_pts), dtype=bool)
    rect = jnp.ones((h, w), dtype=bool)

    mask = lax.dynamic_update_slice(mask, rect, (a, b)).flatten()
    return mask


def get_mask_rect_np(u, h, w):
    grid_pts = int(u.shape[1] ** 0.5)

    a = np.random.randint(0, grid_pts - h + 1, size=())
    b = np.random.randint(0, grid_pts - w + 1, size=())

    mask = np.zeros((grid_pts, grid_pts), dtype=bool)
    mask[a : a + h, b : b + w] = True
    mask = mask.flatten()

    return mask
