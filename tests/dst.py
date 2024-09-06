import sys

sys.path.append("../src")

import jax
import unittest
import numpy as np
import scipy

from functional_autoencoders.util.fft import dst, dstn, idst, idstn


def relative_error(a, b):
    return np.sum((a - b) ** 2) / np.sum(a**2)


eps = 1e-2


dst = jax.jit(dst, static_argnums=(1, 2, 3))
dstn = jax.jit(dstn, static_argnums=(1, 2, 3))


class DST(unittest.TestCase):
    def test_1d_sins(self):
        x = np.linspace(0, 1, 101)[1:-1]
        y = 2 * np.sin(np.pi * x) + 2 * np.sin(np.pi * 3 * x)

        yhat_ours = dst(y, type=1, norm="forward")
        yhat_true = scipy.fft.dst(y, type=1, norm="forward")
        self.assertTrue(relative_error(yhat_true, yhat_ours) < eps)
        self.assertAlmostEqual(yhat_ours[0], 1.0)
        self.assertAlmostEqual(yhat_ours[1], 0.0)
        self.assertAlmostEqual(yhat_ours[2], 1.0)
        self.assertAlmostEqual(yhat_ours[3], 0.0)

    def test_1d_rand(self):
        y = np.random.randn(5001)
        yhat_ours = dst(y, type=1, norm="forward")
        yhat_true = scipy.fft.dst(y, type=1, norm="forward")
        self.assertTrue(relative_error(yhat_true, yhat_ours) < eps)

    def test_2d_rand(self):
        y = np.random.randn(33, 5001)
        yhat_ours = dst(y, type=1, norm="forward")
        yhat_true = scipy.fft.dst(y, type=1, norm="forward")
        self.assertTrue(relative_error(yhat_true, yhat_ours) < eps)

    def test_2d_rand_ortho(self):
        y = np.random.randn(33, 5001)
        yhat_ours = dst(y, type=1, norm="ortho")
        yhat_true = scipy.fft.dst(y, type=1, norm="ortho")
        self.assertTrue(relative_error(yhat_true, yhat_ours) < eps)

    def test_2d_rand_backward(self):
        y = np.random.randn(33, 5001)
        yhat_ours = dst(y, type=1, norm="backward")
        yhat_true = scipy.fft.dst(y, type=1, norm="backward")
        self.assertTrue(relative_error(yhat_true, yhat_ours) < eps)

    def test_dst_self_inverse(self):
        n = 101
        y = np.random.randn(101)
        yhat = dst(y, type=1)
        yhathat = dst(yhat, type=1)
        true = 2 * (n + 1)
        self.assertTrue((np.mean(yhathat / y) - true) / true < eps)

    def test_inverse(self):
        y = np.random.randn(101)
        self.assertTrue(relative_error(y, idst(dst(y, type=1), type=1)) < eps)
        self.assertTrue(
            relative_error(
                y, idst(dst(y, type=1, norm="forward"), type=1, norm="forward")
            )
            < eps
        )


class DSTN(unittest.TestCase):
    def test_dstn_corresponds_with_multiple_dst(self):
        y = np.random.randn(33, 33, 33)
        yhat_true = scipy.fft.dstn(y, type=1, norm="forward")
        yhat_dstn = dstn(y, type=1, norm="forward", axes=(0, 1, 2))
        for axis in [0, 1, 2]:
            y = dst(y, type=1, norm="forward", axis=axis)
        self.assertTrue(relative_error(yhat_true, yhat_dstn) < 1)
        self.assertTrue(relative_error(yhat_true, y) < 1)

    def test_dstn_partial(self):
        y = np.random.randn(32, 33, 33)
        yhat_true = scipy.fft.dstn(y, type=1, norm="forward", axes=(1, 2))
        yhat_ours = dstn(y, type=1, norm="forward", axes=(1, 2))
        self.assertTrue(relative_error(yhat_true, yhat_ours) < 1)

    def test_inverse(self):
        y = np.random.randn(9, 9, 7)
        self.assertTrue(relative_error(y, idstn(dstn(y, type=1), type=1)) < 1)


if __name__ == "__main__":
    unittest.main()
