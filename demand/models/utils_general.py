import argparse
import numpy as np
import demand.models.utils_stat as us
from sklearn.metrics import mean_squared_error


def str2bool(v):
    print(v)
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_nll_by_sample(a_all, b_all, y):
    nll_by_sample = []

    n_samples = a_all.shape[0]

    all_nll = -1 * us.gammapoisson_logpmf(y[:, :, 0], a_all, b_all)

    for n in range(n_samples):
        sample_a_all = a_all[n, :]
        sample_b_all = b_all[n, :]
        sample_y = y[n, :, 0]

        sample_nll = np.mean(-1 * us.gammapoisson_logpmf(sample_y, sample_a_all, sample_b_all))
        nll_by_sample.append(sample_nll)

    total_nll = np.mean(all_nll)

    return np.array(nll_by_sample), total_nll


def get_rmse_by_sample(a_all, b_all, y):
    rmse_by_sample = []

    n_samples = a_all.shape[0]

    all_mean = us.gammapoisson_mean(a_all, b_all)

    for n in range(n_samples):
        sample_mean = all_mean[n, :]
        sample_y = y[n, :, 0]

        sample_rmse = np.sqrt(mean_squared_error(sample_mean, sample_y))
        rmse_by_sample.append(sample_rmse)

    total_rmse = np.sqrt(mean_squared_error(all_mean, y[:,:,0]))

    return np.array(rmse_by_sample), total_rmse

def get_avg_forecast_std_by_sample(a_all, b_all):
    std_by_sample = []

    n_samples = a_all.shape[0]

    all_var = us.gammapoisson_var(a_all, b_all)

    for n in range(n_samples):
        sample_var = all_var[n, :]
        sample_std = np.mean(np.sqrt(sample_var))
        std_by_sample.append(sample_std)

    avg_std = np.mean(np.sqrt(all_var))

    return np.array(std_by_sample), avg_std
