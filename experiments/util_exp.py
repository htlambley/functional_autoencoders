import os
import numpy as np
from functional_autoencoders.util import pickle_load


def get_mse_losses_per_quantity_over_time(data_output_dir, quantity):
    mse_over_runs = {}
    training_times_over_runs = {}
    mse_test_losses_by_quantity_over_time = {}
    for run_idx_str in os.listdir(data_output_dir):
        for quantity_value in os.listdir(os.path.join(data_output_dir, run_idx_str)):
            result = pickle_load(
                os.path.join(
                    data_output_dir, run_idx_str, quantity_value, "data.pickle"
                )
            )

            mse = result["training_results"]["metrics_history"]["MSE (in L^{2})"]
            training_time = result["additional_data"]["training_time"]
            quantity_value = result["additional_data"][quantity]

            if quantity_value not in mse_over_runs:
                mse_over_runs[quantity_value] = []
                training_times_over_runs[quantity_value] = []

            mse_over_runs[quantity_value].append(mse)
            training_times_over_runs[quantity_value].append(training_time)

    for quantity_value in mse_over_runs.keys():
        mse = np.array(mse_over_runs[quantity_value])
        training_times = np.array(training_times_over_runs[quantity_value])

        mse_values_mean = np.mean(mse, axis=0)
        training_times_mean = np.mean(training_times, axis=0)
        t_range = np.linspace(0, training_times_mean, len(mse_values_mean))

        mse_test_losses_by_quantity_over_time[quantity_value] = {
            t: mse for t, mse in zip(t_range, mse_values_mean)
        }

    return mse_test_losses_by_quantity_over_time
