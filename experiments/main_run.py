import sys

sys.path.append(".")
sys.path.append("src/")

import jax
from time import time
from experiments.exp_baseline_comparisons.main import (
    run_baseline_comparisons,
)
from experiments.exp_dirac.main import run_dirac
from experiments.exp_rec_mse_vs_downsample_ratio.main import (
    run_rec_mse_vs_downsample_ratio,
)
from experiments.exp_rec_mse_vs_point_ratio.main import run_rec_mse_vs_point_ratio
from experiments.exp_sde1d.main import run_sde1d
from experiments.exp_sde2d.main import run_sde2d
from experiments.exp_sparse_training.main import run_sparse_training
from experiments.exp_sparse_vs_dense_wall_clock_training.main import (
    run_sparse_vs_dense_wall_clock_training,
)
from experiments.exp_train_vs_inference_wall_clock.main import (
    run_train_vs_inference_wall_clock,
)


def wrap_run(func):
    def wrapped_func(*args, **kwargs):
        try:
            print("*" * 40)
            print(f'Saving to: {kwargs["output_dir"]}')

            start_time = time()
            func(*args, **kwargs)

            print("Done!")
            print(f"Time taken: {(time() - start_time) / 60:.2f} minutes")
            print("*" * 40 + "\n")

        except Exception as e:
            print("Run failed!")
            print(e)

    return wrapped_func


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    start_time = time()

    wrap_run(run_baseline_comparisons)(
        key=key,
        output_dir="tmp/experiments/exp_baseline_comparisons/cnn",
        config_path="experiments/configs/config_cnn.yaml",
        n_runs=5,
        ns_viscosity=1e-4,
        is_darcy=False,
    )

    wrap_run(run_baseline_comparisons)(
        key=key,
        output_dir="tmp/experiments/exp_baseline_comparisons/point",
        config_path="experiments/configs/config_fae.yaml",
        n_runs=5,
        ns_viscosity=1e-4,
        is_darcy=False,
    )

    wrap_run(run_dirac)(
        key=key,
        output_dir="tmp/experiments/exp_dirac/fae",
        config_path="experiments/configs/config_dirac_fae.yaml",
        n_runs=50,
        resolutions=(8, 16, 32, 64, 128),
    )

    wrap_run(run_dirac)(
        key=key,
        output_dir="tmp/experiments/exp_dirac/vano",
        config_path="experiments/configs/config_dirac_vano.yaml",
        n_runs=50,
        resolutions=(8, 16, 32, 64, 128),
    )

    wrap_run(run_rec_mse_vs_downsample_ratio)(
        key=key,
        output_dir="tmp/experiments/exp_rec_mse_vs_downsample_ratio",
        config_path="experiments/configs/config_fae.yaml",
        n_runs=5,
        ns_viscosity=1e-4,
        downsample_ratios=(1, 2, 4, 8),
        enc_point_ratio_train=-1,
    )

    wrap_run(run_rec_mse_vs_point_ratio)(
        key=key,
        output_dir="tmp/experiments/exp_rec_mse_vs_point_ratio",
        config_path="experiments/configs/config_fae.yaml",
        n_runs=5,
        ns_viscosity=1e-4,
        enc_point_ratio_train_list=(0.1, 0.5, 0.9),
        enc_point_ratio_test_list=(0.1, 0.3, 0.5, 0.7, 0.9),
    )

    wrap_run(run_sde1d)(
        key=key,
        output_dir="tmp/experiments/exp_sde1d",
        config_path="experiments/configs/config_sde1d.yaml",
        theta_list=(0, 25, 10_000),
    )

    wrap_run(run_sde2d)(
        key=key,
        output_dir="tmp/experiments/exp_sde2d",
        config_path="experiments/configs/config_sde2d.yaml",
    )

    wrap_run(run_sparse_training)(
        key=key,
        output_dir="tmp/experiments/sparse_training",
        config_path="experiments/configs/config_fae.yaml",
        ratio_rand_pts_enc=0.3,
        ns_viscosity=1e-4,
        is_darcy=False,
    )

    wrap_run(run_sparse_training)(
        key=key,
        output_dir="tmp/experiments/sparse_training_darcy",
        config_path="experiments/configs/config_fae.yaml",
        ratio_rand_pts_enc=0.3,
        ns_viscosity=1e-4,
        is_darcy=True,
    )

    wrap_run(run_sparse_vs_dense_wall_clock_training)(
        key=key,
        output_dir="tmp/experiments/exp_sparse_vs_dense_wall_clock_training",
        config_path="experiments/configs/config_fae_timing.yaml",
        n_runs=5,
        downscale=2,
        ratio_rand_pts_enc_train_list=(0.1, 0.5, 1),
    )

    wrap_run(run_train_vs_inference_wall_clock)(
        key=key,
        output_dir="tmp/experiments/exp_train_vs_inference_wall_clock",
        config_path="experiments/configs/config_fae.yaml",
        n_runs=5,
        downscale=2,
        ratio_rand_pts_enc_train_list=(0.1, 1),
    )

    print("\n" + "-" * 40 + "\n")
    print(f"Total time taken: {(time() - start_time) / 60:.2f} minutes")
    print("\n" + "-" * 40 + "\n")
