import sys, os, pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

plt.style.use('ggplot')

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def plot_arima(predicted_mean, se_mean, train_ts_temp, test_ts_temp, train_temp, test_temp, countries_dir, continent, country, per_capita_flag):
    # plot historical data using gaussian statistics
    aggs_stats = {}
    for a in [0.2, 0.4, 0.6, 0.8]:
        p = a / 2
        multiplier = st.norm.ppf(1 - p)

        forecast_ci_lower_man = predicted_mean - multiplier * se_mean
        forecast_ci_upper_man = predicted_mean + multiplier * se_mean

        aggs_stats[f'{round((0.5 - (1.0 - a) / 2) * 100)}_per_val'] = forecast_ci_lower_man
        aggs_stats[f'{round((0.5 + (1.0 - a) / 2) * 100)}_per_val'] = forecast_ci_upper_man

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.append(train_ts_temp[-1], test_ts_temp), np.append(train_temp[-1], test_temp),
             color="#962938", label='Unobserved Consumption')
    plt.plot(train_ts_temp, train_temp, color="black", label='Observed Consumption')

    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['10_per_val']),
        np.append(train_temp[-1], aggs_stats['20_per_val']),
        alpha=0.1, label='80% Forecast Cred. Reg.', color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['20_per_val']),
        np.append(train_temp[-1], aggs_stats['30_per_val']),
        alpha=0.3, label='60% Forecast Cred. Reg.', color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['30_per_val']),
        np.append(train_temp[-1], aggs_stats['40_per_val']),
        alpha=0.5, label='40% Forecast Cred. Reg.', color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['40_per_val']),
        np.append(train_temp[-1], aggs_stats['60_per_val']),
        alpha=0.7, label='20% Forecast Cred. Reg.', color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['60_per_val']),
        np.append(train_temp[-1], aggs_stats['70_per_val']),
        alpha=0.5, color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['70_per_val']),
        np.append(train_temp[-1], aggs_stats['80_per_val']),
        alpha=0.3, color='#003366')
    plt.fill_between(
        np.append(train_ts_temp[-1], test_ts_temp),
        np.append(train_temp[-1], aggs_stats['80_per_val']),
        np.append(train_temp[-1], aggs_stats['90_per_val']),
        alpha=0.1, color='#003366')

    bottom, top = plt.ylim()
    plt.plot(np.array([train_ts_temp[-1], train_ts_temp[-1]]), np.array([bottom, top]), color="black",
             linestyle='dashed')

    plt.legend(loc='upper left')
    plt.title(f'Forecast Model Backtest for {country}')

    plt.xlabel(f'Year')

    if per_capita_flag:
        plt.ylabel(f'Electricity Consumption Per Capita (kWh/person)')
    else:
        plt.ylabel(f'Electricity Consumption (GWh)')

    plt.savefig(os.path.join(countries_dir, f'{continent}-{country}-pc-{per_capita_flag}-{train_ts_temp[0]}-{test_ts_temp[-1]}.pdf'))
    plt.close()


    # todo: delete this?
    ####################################
    # temporary code to save outputs for arima

    output_tuple = (predicted_mean, se_mean, train_ts_temp, test_ts_temp, train_temp, test_temp, countries_dir, continent, country, per_capita_flag)
    with open(os.path.join(countries_dir, f'{continent}-{country}-pc-{per_capita_flag}-{train_ts_temp[0]}-{test_ts_temp[-1]}.p'), 'wb') as file:
        pickle.dump(output_tuple, file)

    ####################################