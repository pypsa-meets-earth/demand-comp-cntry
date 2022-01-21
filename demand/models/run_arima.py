import sys, os, argparse, pickle
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
import demand.models.utils_general as ug
from demand.data.load_data import load_data
from demand.data.load_data import lookup_country_name, replace_feat_name_with_formal_name
from demand.visualization.plot_arima_by_country import plot_arima


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run sampler to infer electrification status.')
    parser.add_argument('-ds', '--data_set', default='iea',
                        help='the name of the data set to use. Current options include: iea (example, deprecated)')
    parser.add_argument('-fnt', '--fold_num_test', type=int, default=1,
                        help='the fold number for k-fold cross-validation')
    parser.add_argument('-fnv', '--fold_num_val', type=int, default=0,
                        help='the fold number for k-fold cross-validation')
    parser.add_argument('-nf', '--num_folds', type=int, default=26,
                        help='the number of folds for k-fold cross-validation')
    parser.add_argument('-p', '--p', type=int, default=0,
                        help='the AR param')
    parser.add_argument('-d', '--d', type=int, default=2,
                        help='the difference param')
    parser.add_argument('-q', '--q', type=int, default=0,
                        help='the MA param')
    parser.add_argument('-x', '--x', type=ug.str2bool, nargs='?',
                        const=True, default=False,
                        help='Whether to use exogenous variables (i.e. ARIMAX vs ARIMA). Training works, but foreacsting is still deprecated in this setup.')
    parser.add_argument('-v', '--varma', type=ug.str2bool, nargs='?',
                        const=True, default=False, help='Whether to use VARMA vs ARIMA.')
    parser.add_argument('-lo', '--load_model', type=ug.str2bool, nargs='?',
                        const=True, default=False, help='whether to load a saved model, or run from scratch.')
    parser.add_argument('-od', '--output_dir',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'out', 'arima'),
                        help='the output directory, before modifications')
    parser.add_argument('-tid', '--task_id', default="0",
                        help='prepend the SLURM task id to the output directory, separated by an underscore. (default "")')
    parser.add_argument('-pcf', '--per_capita_flag', type=ug.str2bool, nargs='?',
                        const=True, default=True,
                        help='A flag to determine whether consumption conisdered as absolute values, or on a per capita basis.')
    parser.add_argument('-ft', '--feat_target', default='tot_elec_cons',
                        help='The target feature.'
                             'Should be mutually exclusive with feats_complete_req, feats_nans_allowed.'
                             'Available options are listed in the arg --all_feats'
                        )
    parser.add_argument('-fcr', '--feats_complete_req', default='["pop", "gdp"]',
                        help='Features that are required, and timeseries samples will only be included for complete'
                             'data. Should be mutually exclusive with feat_target and feats_nans_allowed.'
                             'Available options are listed in the arg --all_feats'
                        )
    parser.add_argument('-fna', '--feats_nans_allowed',
                        default='["elec_prod", "coal_prod", "coal_netexp", "ng_prod", "ng_netexp", "oil_prod", "oil_netexp", "renew_prod", "battle_deaths", "cdd", "hdd"]',
                        help='Features that are required, and timeseries samples will allow nan values in'
                             'data. Should be mutually exclusive with feats_complete_req'
                             'Available options are listed in the arg --all_feats')
    parser.add_argument('-af', '--all_feats',
                        default='["pop", "gdp", "temp", "rain", "res_elec_cons", "ci_elec_cons", "tot_elec_cons", "elec_prod", "elec_netexp", "coal_prod", "coal_netexp", "ng_prod", "ng_netexp", "oil_prod", "oil_netexp", "renew_prod", "renew_netexp", "battle_deaths", "cdd", "hdd"]',
                        help='All features to add')
    parser.add_argument('-st', '--scale_type',
                        default='minmax',
                        help='Type of scaling to apply. Options include: "minmax" and "zscore"')
    parser.add_argument('-nzan',
                        '--nans_to_zeros_after_norm',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag to convert nans to zeros. Currently, this needs to be "True".')
    parser.add_argument('-ypr', '--years_pre', type=int, default=15,
                        help='years for training')
    parser.add_argument('-ypo', '--years_post', type=int, default=15,
                        help='years for forecasting')

    return parser.parse_args(args)


def norm_nll(data, mu, sigma):
    ll = np.mean(-np.log(sigma * np.sqrt(2 * np.pi)) - (data - mu) ** 2 / (2 * sigma ** 2))
    return -ll


def augment_args(args, orig_args):

    def diff_of_lists(li1, li2):
        return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

    keys_in_orig_but_not_new = diff_of_lists(orig_args.keys(), args.keys())
    for key in keys_in_orig_but_not_new:
        args[key] = orig_args[key]

    return args

def run_arima(args):
    orig_args = parse_args([])
    orig_args = orig_args.__dict__

    # inherit args from lower level
    args = augment_args(args, orig_args)

    # load the appropriate data
    if args["data_set"] == 'iea':

        (dataset, dataset_norm,
         (train_x, train_y, val_x, val_y, test_x, test_y),
         (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts),
         (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry),
         (_, _, _),
         (_, _, _),
         (feat_titles, train_x_unnorm, test_x_unnorm, x0_std, all_cons)
         ) = \
            load_data(fold_num_test=int(args["fold_num_test"]),
                      fold_num_val=int(args["fold_num_val"]),
                      normalize_hist_cons_in_x=False,
                      num_folds=int(args["num_folds"]),
                      scale_type=args["scale_type"],
                      per_capita_flag=args["per_capita_flag"],
                      all_feats=args['all_feats'],
                      feats_complete_req=args['feats_complete_req'],
                      feats_nans_allowed=args['feats_nans_allowed'],
                      nans_to_zeros_after_norm=args['nans_to_zeros_after_norm'],
                      years_pre=int(args['years_pre']),
                      years_post=int(args['years_post']))

    else:
        raise ValueError(f'args.data_set not valid, given as {args["data_set"]}')

    cons_data = test_x[:, :, 0]
    all_cons_data_avail = np.all((cons_data > np.finfo(np.float32).eps), axis=1)

    train = test_x[all_cons_data_avail, :, :]
    test = test_y[all_cons_data_avail, :]
    train_ts = test_x_ts[all_cons_data_avail, :, :]
    test_ts = test_y_ts[all_cons_data_avail, :, :]
    train_cntry = test_x_cntry[all_cons_data_avail, :, :]
    test_cntry = test_y_cntry[all_cons_data_avail, :, :]
    train_unnorm = test_x_unnorm[all_cons_data_avail, :, :]

    mean_test_cons = np.mean(test)

    test_x_cntry_flat = test_x_cntry[all_cons_data_avail, :, 0].flatten()
    acr = np.unique(test_x_cntry_flat)[0]
    country, continent = lookup_country_name(acr)

    # format output
    if args['task_id'] == '0':  # if this is being run locally
        if not args['varma']:
            out_basename = f"ARIMA{'X' if args['x'] else ''}(" + f"{args['p']}," + f"{args['d']}," + f"{args['q']})" + f'_{country}' + f'_pc{args["per_capita_flag"]}'
            output_dir = os.path.join(args["output_dir"] + f"{'X' if args['x'] else ''}", out_basename)
        else:
            out_basename = f"VARMA(" + f"{args['p']}," + f"{args['q']})" + f'_{country}' + f'_pc{args["per_capita_flag"]}'
            output_dir = os.path.join(os.path.split(args["output_dir"])[0], 'varma', out_basename)
    else:  # if this is being run on a slurm environment
        output_dir = os.path.join(args["output_dir"] + f"{'X' if args['x'] else ''}")
    os.makedirs(output_dir, exist_ok=True)

    # with the independent model ARIMA paradigm, we actually just train on x and test on y.
    n_samples = train.shape[0]
    n_to_forecast = test.shape[1]

    main_feat_ind = 0

    # iterate through the samples
    nll_list = []
    rmse_list = []
    rmse_frac_of_mean_list = []
    mean_forecast_std_list = []
    mean_forecast_std_frac_of_mean_list = []

    frac_of_tot_available = []
    for i in range(n_samples):
        endog_varma_temp = np.array(train_unnorm[i, :, :], dtype=float)
        endog_temp = np.array(train_unnorm[i, :, main_feat_ind], dtype=float)
        exog_temp = np.array(np.delete(train_unnorm, main_feat_ind, 2)[i, :, :], dtype=float)
        test_temp = test[i, :]

        train_ts_temp = np.array(train_ts[i, :, main_feat_ind], dtype=float)
        test_ts_temp = test_ts[i, :, :].flatten()

        train_mask = np.invert((endog_temp < np.finfo(np.float32).eps) & (endog_temp > -np.finfo(np.float32).eps))
        endog_temp = endog_temp[train_mask]
        train_ts_temp = train_ts_temp[train_mask]

        if args['varma']:
            model = VARMAX(endog_varma_temp, order=(int(args['p']), int(args['q'])),
                           measurement_error=True,
                           initialization='approximate_diffuse')
        elif args['x']:
            model = SARIMAX(endog_temp, exog=exog_temp, order=(int(args['p']), int(args['d']), int(args['q'])))
        else:
            model = ARIMA(endog_temp, order=(int(args['p']), int(args['d']), int(args['q'])))


        try:
            model_fit = model.fit()
            model_forecast = model_fit.get_forecast(steps=n_to_forecast)

            if args['varma']:
                predicted_mean = model_forecast.predicted_mean[:, main_feat_ind]
                se_mean = model_forecast.se_mean[:, main_feat_ind]
            else:
                predicted_mean = model_forecast.predicted_mean
                se_mean = model_forecast.se_mean

            nll = norm_nll(test_temp, predicted_mean, se_mean)
            rmse = np.sqrt(np.mean((test_temp - predicted_mean) ** 2))

            countries_dir = os.path.join(output_dir, 'countries_prob_preds')
            os.makedirs(countries_dir, exist_ok=True)

            plot_arima(predicted_mean, se_mean, train_ts_temp, test_ts_temp, endog_temp, test_temp, countries_dir, continent,
                       country, args['per_capita_flag'])

            mean_forecast_std = np.mean(se_mean)

        except:
            nll = np.nan
            rmse = np.nan
            mean_forecast_std = np.nan


        nll_list.append(nll)
        rmse_list.append(rmse)
        mean_forecast_std_list.append(mean_forecast_std)
        frac_of_tot_available.append(np.sum(train_mask)/len(train_mask))
        rmse_frac_of_mean_list.append(rmse/mean_test_cons)
        mean_forecast_std_frac_of_mean_list.append(mean_forecast_std/mean_test_cons)


    # save output_data
    output_dict = {
        'acr': acr,
        'country': country,
        'continent': continent,
        'args': args,
        'x': args['x'],
        'fold_num_test': args['fold_num_test'],
        'fold_num_val': args['fold_num_val'],
        'train': train,
        'test': test,
        'train_ts': train_ts,
        'test_ts': test_ts,
        'train_cntry': train_cntry,
        'test_cntry': test_cntry,
        'feat_titles': feat_titles,
        'train_unnorm': train_unnorm,
        'x0_std': x0_std,
        'nll_list': nll_list,
        'rmse_list': rmse_list,
        'rmse_frac_of_mean_list': rmse_frac_of_mean_list,
        'mean_forecast_std_list': mean_forecast_std_list,
        'mean_forecast_std_frac_of_mean_list': mean_forecast_std_frac_of_mean_list,
        'nll_mean': np.mean(nll_list),
        'rmse_mean': np.mean(rmse_list),
        'frac_of_tot_available': frac_of_tot_available,
        'n_samples': n_samples
    }

    pickle.dump(output_dict, open(os.path.join(output_dir, 'output_dict.p'), "wb"))


def main(args):
    args = parse_args(args)
    args = args.__dict__

    run_arima(args)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
