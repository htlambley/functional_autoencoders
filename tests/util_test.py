import sys

sys.path.append("../src")

import unittest
import jax
import jax.numpy as jnp
import numpy as np

from functional_autoencoders.util import get_transition_matrix, bucket_data
from functional_autoencoders.train.metrics import maximum_mean_discrepancy, functional_squared_exponential


class TestTransitionMatrix(unittest.TestCase):
    def test_simple_transition(self):
        """Test a simple deterministic transition sequence."""
        # Sequence: 0 -> 1 -> 2 -> 0 -> 1 -> 2
        u_bucket = jnp.array([[0, 1, 2, 0, 1, 2]], dtype=jnp.float32)
        n = 3
        P = get_transition_matrix(u_bucket, n)

        # Expected: 0->1 (count 2), 1->2 (count 2), 2->0 (count 1)
        # Row 0: [0, 1, 0] (normalized)
        # Row 1: [0, 0, 1] (normalized)
        # Row 2: [1, 0, 0] (normalized)
        expected = jnp.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])

        np.testing.assert_array_almost_equal(P, expected, decimal=5)

    def test_stationary_distribution(self):
        """Test when all transitions are from same state."""
        u_bucket = jnp.array([[1, 1, 1, 1]], dtype=jnp.float32)
        n = 3
        P = get_transition_matrix(u_bucket, n)

        # All transitions are 1 -> 1
        expected_row_1 = jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(P[0, 1, :], expected_row_1, decimal=5)

    def test_empty_row_handling(self):
        """Test handling of rows with no outgoing transitions."""
        # State 2 has no outgoing transitions
        u_bucket = jnp.array([[0, 1, 0, 1]], dtype=jnp.float32)
        n = 3
        P = get_transition_matrix(u_bucket, n)

        # Row 2 should be uniform (1/3, 1/3, 1/3)
        np.testing.assert_array_almost_equal(
            P[0, 2, :], jnp.ones(3) / 3, decimal=5
        )

    def test_batched(self):
        """Test batched input."""
        u_bucket = jnp.array(
            [[0, 1, 0, 1], [1, 2, 1, 2]], dtype=jnp.float32
        )
        n = 3
        P = get_transition_matrix(u_bucket, n)

        self.assertEqual(P.shape, (2, 3, 3))


class TestBucketData(unittest.TestCase):
    def test_simple_bucketing(self):
        """Test simple 2x2 bucketing."""
        x_locs = jnp.array([0.0, 0.5, 1.0])
        y_locs = jnp.array([0.0, 0.5, 1.0])

        # Test points in each quadrant
        u = jnp.array(
            [
                [
                    [0.25, 0.25],  # Bottom-left: bucket 0
                    [0.75, 0.25],  # Bottom-right: bucket 1
                    [0.25, 0.75],  # Top-left: bucket 2
                    [0.75, 0.75],  # Top-right: bucket 3
                ]
            ]
        )

        result = bucket_data(u, x_locs, y_locs)
        expected = jnp.array([[0, 1, 2, 3]], dtype=jnp.float32)
        np.testing.assert_array_equal(result, expected)

    def test_boundary_handling(self):
        """Test that boundaries are handled correctly."""
        x_locs = jnp.array([0.0, 0.5, 1.0])
        y_locs = jnp.array([0.0, 0.5, 1.0])

        # Point exactly on boundary (0.5, 0.5) should go to bucket 3 (top-right)
        # because bins are (x_locs[j], x_locs[j+1]]
        u = jnp.array([[[0.5, 0.5]]])
        result = bucket_data(u, x_locs, y_locs)

        # Exact boundary behavior depends on searchsorted side='right'
        # 0.5 with side='right' in [0, 0.5, 1.0] returns 2, so idx = 1 (second bin)
        expected = jnp.array([[3]], dtype=jnp.float32)
        np.testing.assert_array_equal(result, expected)

    def test_out_of_bounds(self):
        """Test out-of-bounds points are marked as -1."""
        x_locs = jnp.array([0.0, 0.5, 1.0])
        y_locs = jnp.array([0.0, 0.5, 1.0])

        u = jnp.array([[[-0.1, 0.5], [1.1, 0.5]]])
        result = bucket_data(u, x_locs, y_locs)
        expected = jnp.array([[-1, -1]], dtype=jnp.float32)
        np.testing.assert_array_equal(result, expected)

    def test_batched(self):
        """Test batched input."""
        x_locs = jnp.array([0.0, 0.5, 1.0])
        y_locs = jnp.array([0.0, 0.5, 1.0])

        u = jnp.array(
            [
                [[0.25, 0.25], [0.75, 0.75]],
                [[0.75, 0.25], [0.25, 0.75]],
            ]
        )
        result = bucket_data(u, x_locs, y_locs)
        expected = jnp.array([[0, 3], [1, 2]], dtype=jnp.float32)
        np.testing.assert_array_equal(result, expected)


class TestMMD(unittest.TestCase):
    def test_identical_distributions(self):
        """MMD between identical distributions should be close to 0."""
        key = jax.random.PRNGKey(0)
        u = jax.random.normal(key, (32, 10, 1))
        x = jnp.linspace(0, 1, 10).reshape(-1, 1)

        kernel = functional_squared_exponential(sigma=1.0)
        mmd = maximum_mean_discrepancy(u, u, x, kernel)

        self.assertAlmostEqual(float(mmd), 0.0, places=4)

    def test_different_distributions(self):
        """MMD between different distributions should be > 0."""
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        u = jax.random.normal(key1, (32, 10, 1))
        v = jax.random.normal(key2, (32, 10, 1)) + 2.0  # Shifted distribution
        x = jnp.linspace(0, 1, 10).reshape(-1, 1)

        kernel = functional_squared_exponential(sigma=1.0)
        mmd = maximum_mean_discrepancy(u, v, x, kernel)

        self.assertGreater(float(mmd), 0.0)

    def test_symmetry(self):
        """MMD should be symmetric: MMD(u, v) == MMD(v, u)."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        u = jax.random.normal(key1, (16, 8, 1))
        v = jax.random.normal(key2, (16, 8, 1)) + 1.0
        x = jnp.linspace(0, 1, 8).reshape(-1, 1)

        kernel = functional_squared_exponential(sigma=1.0)
        mmd_uv = maximum_mean_discrepancy(u, v, x, kernel)
        mmd_vu = maximum_mean_discrepancy(v, u, x, kernel)

        self.assertAlmostEqual(float(mmd_uv), float(mmd_vu), places=5)


if __name__ == "__main__":
    unittest.main()
