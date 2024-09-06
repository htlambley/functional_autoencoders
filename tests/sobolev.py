import sys

sys.path.append("../src")

import unittest
import jax.numpy as jnp

from functional_autoencoders.domains.grid import ZeroBoundaryConditions


def relative_error(a, b):
    return jnp.sum((a - b) ** 2) / jnp.sum(a**2)


class SobolevNorm(unittest.TestCase):
    def test_sin(self):
        n_pts = 100
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        pts = jnp.reshape(pts, (1, -1, 1))
        self.assertAlmostEqual(
            ZeroBoundaryConditions(0).squared_norm(x, pts), 1.0, places=3
        )
        self.assertAlmostEqual(
            ZeroBoundaryConditions(1).squared_norm(x, pts), 2.0, places=3
        )
        self.assertAlmostEqual(
            ZeroBoundaryConditions(2).squared_norm(x, pts), 4.0, places=3
        )

    def test_batched(self):
        n_pts = 100
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        x = jnp.tile(x, (32, 1, 1))
        pts = jnp.reshape(pts, (1, -1, 1))
        pts = jnp.tile(pts, (32, 1, 1))
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(0).squared_norm(x, pts) - 1.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(1).squared_norm(x, pts) - 2.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(2).squared_norm(x, pts) - 4.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )

    def test_2d_out_dim(self):
        n_pts = 100
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        x = jnp.tile(x, (32, 1, 2))
        pts = jnp.reshape(pts, (1, -1, 1))
        pts = jnp.tile(pts, (32, 1, 1))
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(0).squared_norm(x, pts) - 2.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(1).squared_norm(x, pts) - 4.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )


class SobolevInnerProd(unittest.TestCase):
    def test_sin(self):
        n_pts = 100
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        pts = jnp.reshape(pts, (1, -1, 1))
        self.assertAlmostEqual(
            ZeroBoundaryConditions(0).inner_product(x, x, pts), 1.0, places=3
        )
        self.assertAlmostEqual(
            ZeroBoundaryConditions(1).inner_product(x, x, pts), 2.0, places=3
        )
        self.assertAlmostEqual(
            ZeroBoundaryConditions(2).inner_product(x, x, pts), 4.0, places=3
        )

    def test_2d_out_dim(self):
        n_pts = 100
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        x = jnp.tile(x, (32, 1, 2))
        pts = jnp.reshape(pts, (1, -1, 1))
        pts = jnp.tile(pts, (32, 1, 1))
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(0).inner_product(x, x, pts)
                - 2.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )
        self.assertAlmostEqual(
            jnp.sum(
                ZeroBoundaryConditions(1).inner_product(x, x, pts)
                - 4.0 * jnp.ones((32,))
            ),
            0.0,
            places=3,
        )

    def test_zero_inner_prod(self):
        n_pts = 101
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        y = jnp.zeros_like(x)
        pts = jnp.reshape(pts, (1, -1, 1))
        self.assertAlmostEqual(
            ZeroBoundaryConditions(1.1).inner_product(x, y, pts),
            0.0,
            places=3,
        )

    def test_inner_prod_is_norm(self):
        n_pts = 101
        pts = jnp.linspace(0, 1, n_pts + 2)[1:-1]
        x = jnp.sqrt(2) * jnp.sin(jnp.pi * pts)
        x = jnp.reshape(x, (1, -1, 1))
        pts = jnp.reshape(pts, (1, -1, 1))
        domain = ZeroBoundaryConditions(-0.9)
        self.assertAlmostEqual(
            domain.inner_product(x, x, pts),
            domain.squared_norm(x, pts),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
