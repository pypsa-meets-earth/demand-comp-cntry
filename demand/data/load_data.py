import sys, os, argparse, pickle, json, hashlib, copy
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

import demand.models.utils_general as ug
from demand.utils.cache import cached_with_io


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This function returns organized raw data by ISO country code before and after normalization. '
                    'Users are given the option to specify whether consumption data is organized by '
                    'country-wide values, or by per capita consumption values. Users are also given the'
                    'ability to specify which features are required without missing values, and which '
                    'features are allowed to be given with missing values.')

    # static
    # download_data.py
    parser.add_argument('-cp', '--cntry_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data',
                                             'raw', 'country-and-continent-codes-list-csv_csv',
                                             'country-and-continent-codes-list-csv_csv.csv'),
                        help='path to lookup tables for countries.')

    # 1960-2019
    # download_data.py
    # https://data.worldbank.org/indicator/SP.POP.TOTL
    parser.add_argument('-pp', '--pop_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'wb_pop',
                                             'API_SP.POP.TOTL_DS2_en_excel_v2_10058049_clean.csv'),
                        help='path to raw population data.')

    # 1960-2018
    # download_data.py
    parser.add_argument('-gp', '--gdp_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw', 'GDPpc-wb-1960-2018',
                                             'GDPpc-wb-1960-2018.csv'),
                        help='path to raw gdp data.')

    # 1970-2018
    # download_data.py
    parser.add_argument('-gps', '--gdp_path_somalia',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'UNdata_Export_20201006_214431107_SomaliaGDPpc',
                                             'UNdata_Export_20201006_214431107_SomaliaGDPpc.csv'),
                        help='path to raw gdp data for somalia.')

    # 1961-2016
    # download_data.py
    parser.add_argument('-tp', '--temp_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'temperature_1960-2016',
                                             'temperature_1960-2016.xlsx'),
                        help='path to raw temperature data.')

    # 1961-2016
    # download_data.py
    parser.add_argument('-rp', '--rain_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'rainfall_1960-2016',
                                             'rainfall_1960-2016.xlsx'),
                        help='path to raw rainfall data.')

    # 1971-2018
    # download_data.py
    parser.add_argument('-isp', '--iea_wes_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'iea_wes_2020-68578195-en',
                                             'WBES-2020-1-EN-20210318T100006.csv'),
                        help='path to raw iea world energy statistics data.')

    # 1971-2018
    # download_data.py
    parser.add_argument('-ibp', '--iea_web_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'iea_web_2020-cde01922-en',
                                             'WBAL-2020-1-EN-20210301T100458.csv'),
                        help='path to raw iea world energy balances data.')

    # 2010-2019 - not currently using this
    parser.add_argument('-cip', '--cdd_iea_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'CDD_18_IEA_20210406',
                                             'Weatherforenergytracker-highlights-CDD18-daily-annual_w_iso.xlsx'),
                        help='CDD 18degC data from the IEA.')

    # 2010-2019 - not currently using this
    parser.add_argument('-hip', '--hdd_iea_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'HDD_18_IEA_20210406',
                                             'Weatherforenergytracker-highlights-HDD18-daily-annual_w_iso.xlsx'),
                        help='HDD 18degC data from the IEA.')

    # 195X-2013
    parser.add_argument('-chp', '--cdd_hdd_atalla_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'CDD_HDD_18_Atalla_20210406',
                                             '1-s2.0-S0360544217318388-mmc1_w_iso.xlsx'),
                        help='CDD and HDD 18degC data from the Atalla et al.')

    # battle_deaths_path
    # 1989-2019
    parser.add_argument('-bdp', '--battle_deaths_path',
                        default=os.path.join(os.environ.get("PROJECT_ROOT"), 'data', 'raw',
                                             'API_VC.BTL.DETH_DS2_en_csv_v2_2167203',
                                             'API_VC.BTL.DETH_DS2_en_csv_v2_2167203.csv'),
                        help='wb battle deaths data path.')

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

    parser.add_argument('-pcf', '--per_capita_flag', default=True,
                        help='A flag to determine whether consumption conisdered as absolute values, or on a per capita basis.')

    parser.add_argument('-fnt', '--fold_num_test', type=int, default=2,
                        help='the fold number for k-fold cross-validation')

    parser.add_argument('-fnv', '--fold_num_val', type=int, default=0,
                        help='the fold number for k-fold cross-validation')

    parser.add_argument('-nf', '--num_folds', type=int, default=26,
                        help='the number of folds for k-fold cross-validation')

    parser.add_argument('-ypr', '--years_pre', type=int, default=15,
                        help='years for training')

    parser.add_argument('-ypo', '--years_post', type=int, default=15,
                        help='years for forecasting')

    parser.add_argument('-st', '--scale_type',
                        default='minmax',
                        help='Type of scaling to apply. Options include: "minmax" and "zscore")')

    parser.add_argument('-nzan',
                        '--nans_to_zeros_after_norm',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag to convert nans to zeros. Currently, this needs to be "True".')

    parser.add_argument('-rfg1',
                        '--remove_feat_groups_of_1',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag whether to include data augmentation by removing feat groups of 1')

    parser.add_argument('-rfg2',
                        '--remove_feat_groups_of_2',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag whether to include data augmentation by removing feat groups of 2')

    parser.add_argument('-rfg3',
                        '--remove_feat_groups_of_3',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing feat groups of 3')

    parser.add_argument('-rfg4',
                        '--remove_feat_groups_of_4',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing feat groups of 4')

    parser.add_argument('-rfg5',
                        '--remove_feat_groups_of_5',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing feat groups of 5')

    parser.add_argument('-rtg1',
                        '--remove_timestep_groups_of_1',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag whether to include data augmentation by removing timestep groups of 1')

    parser.add_argument('-rtg2',
                        '--remove_timestep_groups_of_2',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='A flag whether to include data augmentation by removing timestep groups of 2')

    parser.add_argument('-rtg3',
                        '--remove_timestep_groups_of_3',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing timestep groups of 3')

    parser.add_argument('-rtg4',
                        '--remove_timestep_groups_of_4',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing timestep groups of 4')

    parser.add_argument('-rtg5',
                        '--remove_timestep_groups_of_5',
                        type=ug.str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help='A flag whether to include data augmentation by removing timestep groups of 5')

    return parser.parse_args(args)


def lookup_country_name(code):
    args = parse_args([])

    filtered_df = pd.read_csv(args.cntry_path)
    filtered_df['Country_Name_short'] = np.array([c.split(',')[0].split('(')[0] for c in filtered_df['Country_Name']])

    filtered_df.loc[
        filtered_df['Country_Name'] == 'Congo, Republic of the', 'Country_Name_short'] = 'Congo, Republic of the'
    filtered_df.loc[
        filtered_df[
            'Country_Name'] == 'Congo, Democratic Republic of the', 'Country_Name_short'] = 'Congo, Democratic Republic of the'

    filtered_df = filtered_df[filtered_df['Three_Letter_Country_Code'] == code]

    country_name = filtered_df['Country_Name_short'].values[0]
    continent_name = filtered_df['Continent_Name'].values[0]

    return country_name, continent_name


def lookup_country_name_from_array(countries_acronyms):
    country_names = []
    continent_names = []

    for acr in countries_acronyms:
        country_name, continent_name = lookup_country_name(acr)
        country_names.append(country_name)
        continent_names.append(continent_name)

    return np.array(country_names), np.array(continent_names)


def replace_feat_name_with_formal_name(feat_names):
    conversion_dict = {
        'cntry': 'Country',
        'year_ind': 'Year Ind.',
        'pc_elec_cons': 'Elec. Cons. p.c.',
        'tot_elec_cons': 'Elec. Cons.',
        'year': 'Year',
        'pop': 'Pop.',
        'gdp': 'GDP p.c.',
        'temp': 'Avg. Temp.',
        'rain': 'Avg. Rain.',

        'elec_prod': 'Elec. Prod.',
        'coal_prod': 'Coal Prod.',
        'coal_netexp': 'Coal Net Exp.',
        'ng_prod': 'Nat. Gas Prod.',
        'ng_netexp': 'Nat. Gas Net Exp.',
        'oil_prod': 'Oil Prod.',
        'oil_netexp': 'Oil Net Exp.',
        'renew_prod': 'Renew. Prod.',
        'battle_deaths': 'Bat. Deaths',
        'cdd': 'Cool Deg. Days',
        'hdd': 'Heat Deg. Days',

        'mean_grad_tot_elec_cons': 'Elec. Cons.',
        'mean_grad_pc_elec_cons': 'Elec. Cons. p.c.',
        'mean_grad_year': 'Year',
        'mean_grad_pop': 'Pop.',
        'mean_grad_gdp': 'GDP p.c.',
        'mean_grad_temp': 'Avg. \n Temp.',
        'mean_grad_rain': 'Avg. \n Rain.',

        'std_grad_tot_elec_cons': 'Elec. Cons.',
        'std_grad_pc_elec_cons': 'Elec. Cons. p.c.',
        'std_grad_year': 'Year',
        'std_grad_pop': 'Pop.',
        'std_grad_gdp': 'GDP p.c.',
        'std_grad_temp': 'Avg. \n Temp.',
        'std_grad_rain': 'Avg. \n Rain.'
    }
    output_list = []
    for feat_name in feat_names:
        output_list.append(conversion_dict[feat_name])

    return output_list


@cached_with_io
def get_elec_from_wes(wes_input_path):
    # load iea wes data
    iea_wes_df = pd.read_csv(wes_input_path, encoding='ISO-8859-1')

    def extract_wes_prod_and_netexp(iea_wes_fuel_df, fuel_name='fuel_name'):
        iea_wes_fuel_df = iea_wes_fuel_df.drop(['Country', 'PRODUCT', 'Product', 'FLOW', 'TIME', 'Flag Codes', 'Flags'],
                                               axis=1)
        iea_wes_fuel_df = iea_wes_fuel_df.dropna(subset=['Value'])
        iea_wes_fuel_df = iea_wes_fuel_df.rename(
            columns={'ï»¿"COUNTRY"': "country_code", 'Flow': 'flow', 'Time': 'year',
                     'Value': 'value'})

        # filter residential consumption
        res_wes_cons_df = iea_wes_fuel_df[iea_wes_fuel_df['flow'] == 'Residential']
        res_wes_cons_df = res_wes_cons_df[res_wes_cons_df['value'] != 0]
        res_wes_cons_df = res_wes_cons_df.drop(['flow'], axis=1)
        res_wes_cons_df = res_wes_cons_df.rename(columns={'value': f"res_{fuel_name}_cons"})

        # filter final consumption
        tot_wes_cons_df = iea_wes_fuel_df.loc[iea_wes_fuel_df['flow'] == 'Final consumption']
        tot_wes_cons_df = tot_wes_cons_df[tot_wes_cons_df['value'] != 0]
        tot_wes_cons_df = tot_wes_cons_df.drop(['flow'], axis=1)
        tot_wes_cons_df = tot_wes_cons_df.rename(columns={'value': f"tot_{fuel_name}_cons"})

        # # filter c&i consumption
        ci_wes_cons_df = pd.merge(res_wes_cons_df, tot_wes_cons_df, on=["country_code", "year"])
        ci_wes_cons_df[f'ci_{fuel_name}_cons'] = ci_wes_cons_df[f'tot_{fuel_name}_cons'] - ci_wes_cons_df[
            f'res_{fuel_name}_cons']
        ci_wes_cons_df = ci_wes_cons_df.drop([f'tot_{fuel_name}_cons', f'res_{fuel_name}_cons'], axis=1)

        # filter production
        wes_prod_df = iea_wes_fuel_df[iea_wes_fuel_df['flow'] == 'Production']
        wes_prod_df = wes_prod_df[wes_prod_df['value'] != 0]
        wes_prod_df = wes_prod_df.drop(['flow'], axis=1)
        wes_prod_df = wes_prod_df.rename(columns={'value': f"{fuel_name}_prod"})

        # filter imports
        wes_imp_df = iea_wes_fuel_df[iea_wes_fuel_df['flow'] == 'Imports']
        wes_imp_df = wes_imp_df[wes_imp_df['value'] != 0]
        wes_imp_df = wes_imp_df.drop(['flow'], axis=1)
        wes_imp_df = wes_imp_df.rename(columns={'value': f"{fuel_name}_imp"})

        # filter exports
        wes_exp_df = iea_wes_fuel_df[iea_wes_fuel_df['flow'] == 'Exports']
        wes_exp_df = wes_exp_df[wes_exp_df['value'] != 0]
        wes_exp_df = wes_exp_df.drop(['flow'], axis=1)
        wes_exp_df = wes_exp_df.rename(columns={'value': f"{fuel_name}_exp"})

        # calc net exports
        wes_netexp_df = pd.merge(wes_imp_df, wes_exp_df, on=["country_code", "year"])
        wes_netexp_df[f'{fuel_name}_netexp'] = - wes_netexp_df[f'{fuel_name}_exp'] - wes_netexp_df[f'{fuel_name}_imp']
        wes_netexp_df = wes_netexp_df.drop([f'{fuel_name}_exp', f'{fuel_name}_imp'], axis=1)

        return res_wes_cons_df, ci_wes_cons_df, tot_wes_cons_df, wes_prod_df, wes_netexp_df

    iea_elec_df = iea_wes_df[iea_wes_df['Product'] == 'Electricity (GWh)']
    res_elec_cons_df, ci_elec_cons_df, tot_elec_cons_df, elec_prod_df, elec_netexp_df = extract_wes_prod_and_netexp(
        iea_elec_df, fuel_name='elec')

    iea_oil_df = iea_wes_df[iea_wes_df['Product'] == 'Crude oil (kt)']
    _, _, _, oil_prod_df, oil_netexp_df = extract_wes_prod_and_netexp(
        iea_oil_df, fuel_name='oil')

    return res_elec_cons_df, ci_elec_cons_df, tot_elec_cons_df, elec_prod_df, elec_netexp_df, oil_prod_df, oil_netexp_df


@cached_with_io
def get_fuel_production_and_netexp_from_web(web_input_path):
    # get all production and netexports data values for coal, ng, oil, and renewables

    def extract_web_prod_and_netexp(iea_web_fuel_df, fuel_name='fuel_name'):
        # takes in world energy balances (web) dataframe for a specific fuel, and outputs properly formatted
        # dataframes for production and net exports

        # get fuel prod
        iea_web_fuel_prod_df = iea_web_fuel_df[iea_web_fuel_df['FLOW'] == 'INDPROD']
        iea_web_fuel_prod_df = iea_web_fuel_prod_df.drop(
            ["Country", 'ï»¿"UNIT"', 'Unit', 'PRODUCT', 'Product', 'FLOW', 'Flow', 'TIME', 'Flag Codes', 'Flags'],
            axis=1)
        iea_web_fuel_prod_df = iea_web_fuel_prod_df.dropna(subset=['Value'])
        iea_web_fuel_prod_df = iea_web_fuel_prod_df.rename(
            columns={'COUNTRY': "country_code", 'Time': 'year',
                     'Value': f'{fuel_name}_prod'})
        # get fuel imp
        iea_web_fuel_imp_df = iea_web_fuel_df[iea_web_fuel_df['FLOW'] == 'IMPORTS']
        iea_web_fuel_imp_df = iea_web_fuel_imp_df.drop(
            ["Country", 'ï»¿"UNIT"', 'Unit', 'PRODUCT', 'Product', 'FLOW', 'Flow', 'TIME', 'Flag Codes', 'Flags'],
            axis=1)
        iea_web_fuel_imp_df = iea_web_fuel_imp_df.dropna(subset=['Value'])
        iea_web_fuel_imp_df = iea_web_fuel_imp_df.rename(
            columns={'COUNTRY': "country_code", 'Time': 'year',
                     'Value': f'{fuel_name}_imp'})
        # get fuel exp
        iea_web_fuel_exp_df = iea_web_fuel_df[iea_web_fuel_df['FLOW'] == 'EXPORTS']
        iea_web_fuel_exp_df = iea_web_fuel_exp_df.drop(
            ["Country", 'ï»¿"UNIT"', 'Unit', 'PRODUCT', 'Product', 'FLOW', 'Flow', 'TIME', 'Flag Codes', 'Flags'],
            axis=1)
        iea_web_fuel_exp_df = iea_web_fuel_exp_df.dropna(subset=['Value'])
        iea_web_fuel_exp_df = iea_web_fuel_exp_df.rename(
            columns={'COUNTRY': "country_code", 'Time': 'year',
                     'Value': f'{fuel_name}_exp'})
        # get fuel netexp
        iea_web_fuel_netexp_df = pd.merge(iea_web_fuel_imp_df, iea_web_fuel_exp_df,
                                          on=["country_code", "year"])
        iea_web_fuel_netexp_df[f'{fuel_name}_netexp'] = - iea_web_fuel_netexp_df[f'{fuel_name}_exp'] - \
                                                        iea_web_fuel_netexp_df[
                                                            f'{fuel_name}_imp']
        iea_web_fuel_netexp_df = iea_web_fuel_netexp_df.drop([f'{fuel_name}_exp', f'{fuel_name}_imp'], axis=1)

        return iea_web_fuel_prod_df, iea_web_fuel_netexp_df

    # load all web data
    iea_web_df = pd.read_csv(web_input_path, encoding='ISO-8859-1')
    iea_web_df = iea_web_df[iea_web_df['Unit'] == 'TJ']

    # filter for coal data
    iea_web_coal_df = iea_web_df[iea_web_df['PRODUCT'] == 'COAL']
    coal_prod_df, coal_netexp_df = extract_web_prod_and_netexp(iea_web_coal_df, fuel_name='coal')

    # filter for ng data
    iea_web_ng_df = iea_web_df[iea_web_df['PRODUCT'] == 'NATGAS']
    ng_prod_df, ng_netexp_df = extract_web_prod_and_netexp(iea_web_ng_df, fuel_name='ng')

    # # filter for oil data
    # iea_web_oil_df = iea_web_df[iea_web_df['PRODUCT'] == 'TOTPRODS']
    # oil_prod_df, oil_netexp_df = extract_web_prod_and_netexp(iea_web_oil_df, fuel_name='oil')

    # filter for renewables data
    iea_web_renew_df = iea_web_df[iea_web_df['PRODUCT'] == 'MRENEW']
    renew_prod_df, renew_netexp_df = extract_web_prod_and_netexp(iea_web_renew_df, fuel_name='renew')

    return coal_prod_df, coal_netexp_df, ng_prod_df, ng_netexp_df, renew_prod_df, renew_netexp_df


@cached_with_io
def get_pop(input_path):
    pop_df = pd.read_csv(input_path, encoding='ISO-8859-1')
    pop_df = pop_df.melt(id_vars=["Country Code", "Country Name"], var_name="year", value_name="pop")
    pop_df = pop_df.rename(columns={"Country Code": "country_code"})
    pop_df = pop_df.drop(["Country Name"], axis=1)
    pop_df.year = pop_df.year.astype('int64')
    pop_df = pop_df.dropna()
    pop_df = pop_df[pop_df['pop'] > 0.0]

    return pop_df


@cached_with_io
def get_gdp(input_path, input_path_somalia):
    # Indicator Name: GDP per capita (constant 2010 US$)
    gdp_df = pd.read_csv(input_path, encoding='ISO-8859-1', skiprows=4)
    gdp_som_df = pd.read_csv(input_path_somalia, encoding='ISO-8859-1', skiprows=0)

    gdp_df = gdp_df.drop(['Indicator Code', 'Indicator Name', 'Country Name'], axis=1)
    gdp_df = gdp_df.melt(id_vars=['Country Code'], var_name="year", value_name="gdp")
    gdp_df = gdp_df.rename(columns={"Country Code": "country_code"})
    gdp_df.year = gdp_df.year.astype('int64')
    gdp_df = gdp_df.dropna()

    som_years = gdp_som_df.shape[0]
    gdp_som_df['country_code'] = np.repeat('SOM', som_years)
    gdp_som_df['year'] = gdp_som_df['Year'].values
    gdp_som_df['gdp'] = gdp_som_df['Value'].values
    gdp_som_df = gdp_som_df.drop(['Country or Area', 'Year', 'Item', 'Value'], axis=1)

    gdp_df = gdp_df.append(gdp_som_df)

    gdp_df = gdp_df[gdp_df['gdp'] > 0.0]

    return gdp_df


@cached_with_io
def get_temp(input_path):
    temp_df = pd.read_excel(input_path)
    temp_df['Year'] = temp_df['Year'].astype(int)
    temp_df['Country'] = temp_df['Country'].str.strip()
    temp_df['ISO3'] = temp_df['ISO3'].str.strip()
    temp_df = temp_df.rename(
        columns={"Country": "country", "ISO3": "country_code", "Year": "year", "Temperature": "temp"})

    # cleaning problems in the data
    temp_df.loc[
        temp_df['country'] == 'Tanzania', 'country_code'] = 'TZA'

    temp_df = temp_df.drop(['country'], axis=1)

    return temp_df


@cached_with_io
def get_rain(input_path):
    rain_df = pd.read_excel(input_path)
    rain_df['Year'] = rain_df['Year'].astype(float)
    rain_df['Country'] = rain_df['Country'].str.strip()
    rain_df['ISO3'] = rain_df['ISO3'].str.strip()
    rain_df = rain_df.rename(
        columns={"Country": "country", "ISO3": "country_code", "Year": "year", "Rainfall (mm)": "rain"})

    # cleaning problems in the data
    rain_df.loc[
        rain_df['country'] == 'Tanzania', 'country_code'] = 'TZA'

    rain_df = rain_df.drop(['country'], axis=1)

    return rain_df


@cached_with_io
def get_battle_deaths(battle_deaths_path):
    battle_deaths_df = pd.read_csv(battle_deaths_path, encoding='ISO-8859-1', skiprows=4)
    battle_deaths_df = battle_deaths_df.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"], var_name="year",
        value_name="battle_deaths")
    battle_deaths_df = battle_deaths_df.dropna(subset=['battle_deaths'])
    battle_deaths_df = battle_deaths_df.drop(['Country Name', "Indicator Name", "Indicator Code"], axis=1)
    battle_deaths_df = battle_deaths_df.rename(columns={'Country Code': "country_code"})
    battle_deaths_df.year = battle_deaths_df.year.astype('int64')

    return battle_deaths_df


@cached_with_io
def get_hdd_cdd(cdd_iea_path, hdd_iea_path, cdd_hdd_atalla_path, plot_atalla_vs_iea=False):
    # melt cdd_atalla into right format
    cdd_atalla_df = pd.read_excel(cdd_hdd_atalla_path,
                                  sheet_name='t2m.cdd.18C_daily_freq_iso')
    cdd_atalla_df = cdd_atalla_df.melt(id_vars=["Country", "ISO"], var_name="year", value_name="cdd_atalla")
    cdd_atalla_df = cdd_atalla_df.drop(['Country'], axis=1)
    cdd_atalla_df = cdd_atalla_df.rename(columns={"ISO": "country_code"})
    cdd_atalla_df.year = cdd_atalla_df.year.astype('int64')

    # melt hdd_atalla into right format
    hdd_atalla_df = pd.read_excel(cdd_hdd_atalla_path,
                                  sheet_name='T2m.hdd.18C_daily_freq_iso')
    hdd_atalla_df = hdd_atalla_df.melt(id_vars=["Country", "ISO"], var_name="year", value_name="hdd_atalla")
    hdd_atalla_df = hdd_atalla_df.drop(['Country'], axis=1)
    hdd_atalla_df = hdd_atalla_df.rename(columns={"ISO": "country_code"})
    hdd_atalla_df.year = hdd_atalla_df.year.astype('int64')

    if plot_atalla_vs_iea:
        # below is code to plot the atalla et al cdd and hdd data sets with the iea data set. The preliminary
        # conclusion about these two datasets is that they are incompatible. Even though they claim to be producing
        # the same thing, their methodologies are different and they give incongruent data for overlapping years

        # melt cdd_iea into right format
        cdd_iea_df = pd.read_excel(cdd_iea_path)
        cdd_iea_df = cdd_iea_df.melt(id_vars=["Country", "ISO Code"], var_name="year", value_name="cdd_iea")
        cdd_iea_df = cdd_iea_df.drop(['Country'], axis=1)
        cdd_iea_df = cdd_iea_df.rename(columns={"ISO Code": "country_code"})

        # melt hdd_iea into right format
        hdd_iea_df = pd.read_excel(hdd_iea_path)
        hdd_iea_df = hdd_iea_df.melt(id_vars=["Country", "ISO Code"], var_name="year", value_name="hdd_iea")
        hdd_iea_df = hdd_iea_df.drop(['Country'], axis=1)
        hdd_iea_df = hdd_iea_df.rename(columns={"ISO Code": "country_code"})

        # get intersection of country codes for cdd entries
        cdd_countries = cdd_atalla_df['country_code'].unique().tolist()
        cdd_countries.extend(cdd_iea_df['country_code'].unique().tolist())
        cdd_countries = np.unique(cdd_countries)
        cdd_countries_intersect = np.intersect1d(cdd_atalla_df['country_code'].unique(),
                                                 cdd_iea_df['country_code'].unique())

        # get intersection of country codes for cdd entries
        hdd_countries = hdd_atalla_df['country_code'].unique().tolist()
        hdd_countries.extend(hdd_iea_df['country_code'].unique().tolist())
        hdd_countries = np.unique(hdd_countries)
        hdd_countries_intersect = np.intersect1d(hdd_atalla_df['country_code'].unique(),
                                                 hdd_iea_df['country_code'].unique())

        # make cdd out path
        cdd_out_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'out', 'cdd')
        os.makedirs(cdd_out_path, exist_ok=True)

        # make cdd out path
        hdd_out_path = os.path.join(os.environ.get("PROJECT_ROOT"), 'out', 'hdd')
        os.makedirs(hdd_out_path, exist_ok=True)

        # print pngs of plots by country

        for cntry in cdd_countries_intersect:
            x1 = cdd_atalla_df[cdd_atalla_df['country_code'] == cntry]['year'].values
            y1 = cdd_atalla_df[cdd_atalla_df['country_code'] == cntry]['cdd_atalla'].values
            x2 = cdd_iea_df[cdd_iea_df['country_code'] == cntry]['year'].values
            y2 = cdd_iea_df[cdd_iea_df['country_code'] == cntry]['cdd_iea'].values
            plt.figure()
            plt.plot(x1, y1, label='Atalla et al.')
            plt.plot(x2, y2, label='IEA')
            plt.title(f'CDD data for {lookup_country_name(cntry)[0]}')
            out_path = os.path.join(cdd_out_path, f'{cntry}.png')
            plt.savefig(out_path)
            plt.close()

        # print pngs of plots by country
        for cntry in hdd_countries_intersect:
            x1 = hdd_atalla_df[hdd_atalla_df['country_code'] == cntry]['year'].values
            y1 = hdd_atalla_df[hdd_atalla_df['country_code'] == cntry]['hdd_atalla'].values
            x2 = hdd_iea_df[hdd_iea_df['country_code'] == cntry]['year'].values
            y2 = hdd_iea_df[hdd_iea_df['country_code'] == cntry]['hdd_iea'].values
            plt.figure()
            plt.plot(x1, y1, label='Atalla et al.')
            plt.plot(x2, y2, label='IEA')
            plt.title(f'hdd data for {lookup_country_name(cntry)[0]}')
            out_path = os.path.join(hdd_out_path, f'{cntry}.png')
            plt.savefig(out_path)
            plt.close()

    cdd_df = cdd_atalla_df
    hdd_df = hdd_atalla_df
    cdd_df = cdd_df.rename(columns={"cdd_atalla": "cdd"})
    hdd_df = hdd_df.rename(columns={"hdd_atalla": "hdd"})

    return cdd_df, hdd_df


def compile_data(args, all_feats=None, feat_target=None, feats_complete_req=None,
                 feats_nans_allowed=None):
    # Compile data from disparate sources, format them all the same way, and
    # combine them into a single df, df_all. Return a full dictionary of
    # this data as output.
    # Take advantage of caching given parameter settings for faster subsequent runs.
    # Note that any changes to data sources requires caches to be cleared.

    # convert args to hash
    args = parse_args(args)
    args = args.__dict__

    # override with params
    if all_feats == None:
        pass
    else:
        args['all_feats'] = all_feats

    if feat_target == None:
        pass
    else:
        args['feat_target'] = feat_target

    if feats_complete_req == None:
        pass
    else:
        args['feats_complete_req'] = feats_complete_req

    if feats_nans_allowed == None:
        pass
    else:
        args['feats_nans_allowed'] = feats_nans_allowed

    args_string = json.dumps(args)
    args_hash = hashlib.sha256(args_string.encode('utf-8')).hexdigest()
    args_hash_pickle_path = Path(os.path.join(os.environ.get('PROJECT_ROOT'), 'data', 'processed', args_hash + '.p'))

    # save to file, if not already a file
    if not args_hash_pickle_path.is_file():

        # load pop gdp temp rain data
        pop_df = get_pop(args['pop_path'])
        gdp_df = get_gdp(args['gdp_path'], args['gdp_path_somalia'])
        temp_df = get_temp(args['temp_path'])
        rain_df = get_rain(args['rain_path'])

        # load elec an oil data
        res_elec_cons_df, ci_elec_cons_df, tot_elec_cons_df, elec_prod_df, elec_netexp_df, oil_prod_df, oil_netexp_df = get_elec_from_wes(
            args['iea_wes_path'])

        # load coal and ng data
        coal_prod_df, coal_netexp_df, \
        ng_prod_df, ng_netexp_df, \
        renew_prod_df, renew_netexp_df = \
            get_fuel_production_and_netexp_from_web(args['iea_web_path'])

        # load cdd, hdd, and battle deaths data
        cdd_df, hdd_df = get_hdd_cdd(args['cdd_iea_path'], args['hdd_iea_path'], args['cdd_hdd_atalla_path'])
        battle_deaths_df = get_battle_deaths(args['battle_deaths_path'])

        # get rid of country labels for all dfs
        scope = locals()

        # load dfs to merge together
        dfs_all_feats = [feat + '_df' for feat in json.loads(args['all_feats'])]
        df_feat_target = args['feat_target'] + '_df'
        dfs_to_load_complete = [feat + '_df' for feat in json.loads(args['feats_complete_req'])]
        dfs_to_load_nans = [feat + '_df' for feat in json.loads(args['feats_nans_allowed'])]

        # turn into dict of dataframes
        df_dict = {df_name: eval(df_name, scope) for df_name in dfs_all_feats}

        # merge of dataframes, requiring complete data
        df_combination_uid = 't_' + df_feat_target + '_c_'
        for i, df_name in enumerate(dfs_to_load_complete):
            df_combination_uid = df_combination_uid + df_name.replace('_df', '')[0] + df_name.replace('_df', '')[-1]
            if i == 0:
                df_all = eval(df_name, scope)
            else:
                if df_name == 'year_df':
                    continue
                df_all = pd.merge(df_all, eval(df_name, scope), on=["country_code", "year"])

        # merge of dataframes, allowing nan values
        df_combination_uid = df_combination_uid + '_n_'
        for i, df_name in enumerate(dfs_to_load_nans):
            df_all = pd.merge(df_all, eval(df_name, scope), how='left', on=["country_code", "year"])
            df_combination_uid = df_combination_uid + df_name.replace('_df', '')[0] + df_name.replace('_df', '')[-1]

        # add in target, but do not require the column to have complete data
        df_all = pd.merge(df_all, eval(df_feat_target, scope), how='left', on=["country_code", "year"])

        # reorder dataframe columns
        cols = df_all.columns.tolist()
        cols = cols[:2] + cols[-1:] + cols[2:-1]
        df_all = df_all[cols]

        df_dict[df_combination_uid] = df_all
        df_dict['df_all'] = df_all

        # saving dict of dfs
        pickle.dump((df_dict, df_combination_uid), open(args_hash_pickle_path, "wb"), protocol=4)

    else:

        df_dict, df_combination_uid = pickle.load(open(args_hash_pickle_path, "rb"))

    return df_dict, df_combination_uid


def get_target_feat_name(per_capita_flag=True):
    if per_capita_flag:
        target_feat_name = 'pc_elec_cons'
    else:
        target_feat_name = 'tot_elec_cons'

    return target_feat_name


def split_dataset_cons(dataset_norm, target_feat_name=None, fold_num_test=0, fold_num_val=1, num_folds=26, years_pre=15,
                       years_post=15):
    # split a multivariate dataset into train/test sets.
    # The logic here is to loop through every country in the given normalized dataset
    # For a given country, we calculate the number of iterations for which data is available
    # given end- and start-years. We the query for these years.
    # We keep track of fold_num_test and num_folds. Only African countries are added to leave-one-out
    # cross validation folds.

    countries = np.unique(dataset_norm['country_code'].values)

    ######################################
    # populate train, val, and test
    ######################################
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    # loop through each unique country
    # the logic is to track non-african countries separately from african countries.
    # only african countries are ever added to the test splits. Non-african countries are
    # always in "train"
    afr_i = 0
    for c, country in enumerate(countries):

        try:
            country_name, continent_name = lookup_country_name(country)
        except:
            print(f'cannot find country: {country}')
            continue

        # get just the country's data
        data_temp_df = dataset_norm[dataset_norm['country_code'] == country]
        start_year = np.min(data_temp_df['year_orig'].values)
        end_year = np.max(data_temp_df['year_orig'].values)

        # calculate the number of iterations to do for a given country.
        # We assume that all years within the start and end years have
        # entries present. Empirically, looking at the data (for population and gdp)
        # ths is true
        num_iters_train_val_test = end_year - start_year - years_pre - years_post + 2

        # if you don't have enough training years for this country, skip it
        if num_iters_train_val_test < 1:
            continue

        # increment africa counter flag
        increment_africa_counter = False

        # if you do have enough training years, start to define training entries and test entry(ies)
        for i in range(num_iters_train_val_test):

            # define start and end indices
            x_start = start_year + i
            x_end = start_year + years_pre + i
            y_end = start_year + years_pre + years_post + i

            x_temp_df_sample = data_temp_df[(data_temp_df['year_orig'] >= x_start)
                                            & (data_temp_df['year_orig'] < x_end)]

            y_temp_df_sample = data_temp_df[(data_temp_df['year_orig'] >= x_end)
                                            & (data_temp_df['year_orig'] < y_end)]

            # now we need to determine whether the whole y series has consumption data (our target feature)
            # If not, we skip it. If so, then we can use it for training/validation/test. If not,
            if (y_temp_df_sample[target_feat_name] < np.finfo(float).eps).any():
                continue

            if (continent_name == 'Africa'):
                increment_africa_counter = True

            # append into arrays if possible!
            if (continent_name == 'Africa') and (afr_i % num_folds == fold_num_test):
                test_x.append(x_temp_df_sample)
                test_y.append(y_temp_df_sample)
                print(f'added {country_name} to test')

            if (continent_name == 'Africa') and (afr_i % num_folds == fold_num_val):
                val_x.append(x_temp_df_sample)
                val_y.append(y_temp_df_sample)
                print(f'added {country_name} to val')

            if not ((continent_name == 'Africa') and (afr_i % num_folds == fold_num_val)) and \
                    not ((continent_name == 'Africa') and (afr_i % num_folds == fold_num_test)):
                train_x.append(x_temp_df_sample)
                train_y.append(y_temp_df_sample)

        if increment_africa_counter:
            afr_i = afr_i + 1

    ######################################
    # populate run_forward
    # this tries to make forecasts using the latest available data for a given country
    ######################################
    future_x = []
    all_cons = []

    # loop through each unique country
    for c, country in enumerate(countries):

        try:
            lookup_country_name(country)
        except:
            print(f'cannot find country: {country}')
            continue

        # get just the country's data
        data_temp_df = dataset_norm[dataset_norm['country_code'] == country]
        end_year = np.max(data_temp_df['year_orig'].values)
        start_year = end_year - years_pre

        x_temp_df_sample = data_temp_df[
            (data_temp_df['year_orig'] > start_year) & (data_temp_df['year_orig'] <= end_year)]

        future_x.append(x_temp_df_sample)
        all_cons.append(data_temp_df)

    ######################################
    # populate run_hist
    # this looks for historical data for all countries
    ######################################
    # calculate start and end years for hist preds
    # kenya_last_year = np.max(dataset_norm[dataset_norm['country_code'] == 'KEN'].index.get_level_values(0).values)
    # end_year = kenya_last_year - years_post + 1
    # start_year = end_year - years_pre

    countries = np.unique(dataset_norm['country_code'].values)

    hist_x = []

    # loop through each unique country
    for c, country in enumerate(countries):

        try:
            lookup_country_name(country)
        except:
            print(f'cannot find country: {country}')
            continue

        # get just the country's data
        data_temp_df = dataset_norm[dataset_norm['country_code'] == country]
        start_year = np.min(data_temp_df['year_orig'].values)
        end_year = np.max(data_temp_df['year_orig'].values)

        # calculate the number of iterations to do for a given country.
        # We assume that all years within the start and end years have
        # entries present. Empirically, looking at the data (for population and gdp)
        # ths is true
        num_iters_hist = end_year - start_year - years_pre - years_post + 2

        # if you don't have enough training years for this country, skip it
        if num_iters_hist < 1:
            continue

        # if you do have enough training years, start to define training entries and test entry(ies)
        for i in range(num_iters_hist):
            # define start and end indices
            x_start = start_year + i
            x_end = start_year + years_pre + i

            x_temp_df_sample = data_temp_df[(data_temp_df['year_orig'] >= x_start)
                                            & (data_temp_df['year_orig'] < x_end)]

            hist_x.append(x_temp_df_sample)

    # process the dfs to make lists of values and years, which is required for
    # keras input data formats for LSTM models
    def unravel_df_list(df_lists, x_or_y='x', years_pre=15, years_post=15):

        list_to_collapse = []
        list_to_collapse_ts = []
        list_to_collapse_cntry = []

        for df_list in df_lists:

            if (x_or_y == 'x') and (df_list['year_orig'].values.size != years_pre):
                print(f"skipping forecasts for {np.unique(df_list['country_code'].values)}")
                continue

            if (x_or_y == 'y') and (df_list['year_orig'].values.size != years_post):
                print(f"skipping forecasts for {np.unique(df_list['country_code'].values)}")
                continue

            list_to_collapse_ts.append(df_list['year_orig'].values)
            list_to_collapse_cntry.append(df_list['country_code'].values)
            list_to_collapse.append(df_list.drop(columns=['country_code', 'year_orig']).values)

        try:
            collapse_ts = np.array(list_to_collapse_ts, dtype=int)
            collapse_cntry = np.array(list_to_collapse_cntry)
            collapse_ts = collapse_ts.reshape(collapse_ts.shape[0], collapse_ts.shape[1], 1)
            collapse_cntry = collapse_cntry.reshape(collapse_cntry.shape[0], collapse_cntry.shape[1], 1)
            collapse_stacked = np.array(list_to_collapse)
        except:
            print('asdf in index 23491351253')

        # reshape 3d tensor if y, since we are doing a single output type
        if x_or_y == 'y':
            collapse_stacked = collapse_stacked[:, :, 0].reshape(collapse_stacked.shape[0], collapse_stacked.shape[1])

        return collapse_stacked, collapse_ts, collapse_cntry

    print('asdf')

    # reshape to numpy arrays
    train_x, train_x_ts, train_x_cntry = \
        unravel_df_list(train_x,
                        x_or_y='x',
                        years_pre=years_pre,
                        years_post=years_post)
    train_y, train_y_ts, train_y_cntry = \
        unravel_df_list(train_y,
                        x_or_y='y',
                        years_pre=years_pre,
                        years_post=years_post)
    val_x, val_x_ts, val_x_cntry = \
        unravel_df_list(val_x,
                        x_or_y='x',
                        years_pre=years_pre,
                        years_post=years_post)
    val_y, val_y_ts, val_y_cntry = \
        unravel_df_list(val_y,
                        x_or_y='y',
                        years_pre=years_pre,
                        years_post=years_post)
    test_x, test_x_ts, test_x_cntry = \
        unravel_df_list(test_x,
                        x_or_y='x',
                        years_pre=years_pre,
                        years_post=years_post)
    test_y, test_y_ts, test_y_cntry = \
        unravel_df_list(test_y,
                        x_or_y='y',
                        years_pre=years_pre,
                        years_post=years_post)
    future_x, future_x_ts, future_x_cntry = \
        unravel_df_list(future_x,
                        x_or_y='x',
                        years_pre=years_pre,
                        years_post=years_post)
    hist_x, hist_x_ts, hist_x_cntry = \
        unravel_df_list(hist_x,
                        x_or_y='x',
                        years_pre=years_pre,
                        years_post=years_post)

    all_cons = pd.concat(all_cons)

    return (train_x, train_y, val_x, val_y, test_x, test_y), \
           (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts), \
           (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry), \
           (future_x, future_x_ts, future_x_cntry), \
           (hist_x, hist_x_ts, hist_x_cntry), \
           all_cons


def remove_all_noncountry_entities(dataset):
    # remove all non-country entities
    dataset = dataset[dataset['country_code'] != 'WLD']
    dataset = dataset[dataset['country_code'] != 'IBT']
    dataset = dataset[dataset['country_code'] != 'LMY']
    dataset = dataset[dataset['country_code'] != 'MIC']
    dataset = dataset[dataset['country_code'] != 'IBD']
    dataset = dataset[dataset['country_code'] != 'EAR']
    dataset = dataset[dataset['country_code'] != 'LMC']
    dataset = dataset[dataset['country_code'] != 'UMC']
    dataset = dataset[dataset['country_code'] != 'EAS']
    dataset = dataset[dataset['country_code'] != 'LTE']
    dataset = dataset[dataset['country_code'] != 'EAP']
    dataset = dataset[dataset['country_code'] != 'TEA']
    dataset = dataset[dataset['country_code'] != 'TSA']
    dataset = dataset[dataset['country_code'] != 'SAS']
    dataset = dataset[dataset['country_code'] != 'IDA']
    dataset = dataset[dataset['country_code'] != 'OED']
    dataset = dataset[dataset['country_code'] != 'HIC']
    dataset = dataset[dataset['country_code'] != 'PST']
    dataset = dataset[dataset['country_code'] != 'SSF']
    dataset = dataset[dataset['country_code'] != 'TSS']
    dataset = dataset[dataset['country_code'] != 'SSA']
    dataset = dataset[dataset['country_code'] != 'IDX']
    dataset = dataset[dataset['country_code'] != 'LDC']
    dataset = dataset[dataset['country_code'] != 'ECS']
    dataset = dataset[dataset['country_code'] != 'PRE']
    dataset = dataset[dataset['country_code'] != 'HPC']
    dataset = dataset[dataset['country_code'] != 'LIC']
    dataset = dataset[dataset['country_code'] != 'LCN']
    dataset = dataset[dataset['country_code'] != 'LAC']
    dataset = dataset[dataset['country_code'] != 'TLA']
    dataset = dataset[dataset['country_code'] != 'IDB']
    dataset = dataset[dataset['country_code'] != 'EUU']
    dataset = dataset[dataset['country_code'] != 'FCS']
    dataset = dataset[dataset['country_code'] != 'TEC']
    dataset = dataset[dataset['country_code'] != 'MEA']
    dataset = dataset[dataset['country_code'] != 'ECA']
    dataset = dataset[dataset['country_code'] != 'ARB']
    dataset = dataset[dataset['country_code'] != 'MNA']
    dataset = dataset[dataset['country_code'] != 'TMN']
    dataset = dataset[dataset['country_code'] != 'NAC']
    dataset = dataset[dataset['country_code'] != 'EMU']
    dataset = dataset[dataset['country_code'] != 'CEB']

    return dataset


# scale dataset
def scale_data(dataset,
               scale_type='minmax',
               per_capita_flag=True,
               feats_complete_req=None,
               feats_nans_allowed=None,
               nans_to_zeros_after_norm=True
               ):
    dataset = dataset.set_index([dataset['year'], dataset['country_code']])
    dataset = remove_all_noncountry_entities(dataset)

    target_feat_name = get_target_feat_name(per_capita_flag=per_capita_flag)

    columns = ['country_code']
    columns.append(target_feat_name)
    columns.extend(json.loads(feats_complete_req))
    columns.extend(json.loads(feats_nans_allowed))

    # adjust consumption for per capita values
    if per_capita_flag:
        dataset[target_feat_name] = dataset['tot_elec_cons'] / dataset['pop'] * 1e6

    # reorganize and filter columns
    dataset = dataset[columns]

    # scaling
    dataset_values = dataset.values
    if scale_type == 'minmax':
        scaler = MinMaxScaler((1.0, 2.0))
    else:
        scaler = StandardScaler()  # scale all auxiliary data values
    scaler = scaler.fit(dataset.values[:, 2:])
    dataset_values[:, 2:] = scaler.transform(dataset_values[:, 2:])

    # defining a z-scored, scaled version of the data
    dataset_norm = pd.DataFrame(data=dataset_values,
                                columns=columns)
    dataset_norm = dataset_norm.set_index(dataset.index)
    dataset_norm['year_orig'] = dataset.index.get_level_values(0).values
    dataset_norm.sort_values(by='year_orig', ascending=True)

    # fill nans with zeros if specified
    if nans_to_zeros_after_norm:
        dataset_norm = dataset_norm.fillna(0)

    return dataset, dataset_norm, scaler


def augment_by_removing_feats(x,
                              y,
                              x_ts,
                              y_ts,
                              x_cntry,
                              y_cntry,
                              remove_feat_groups_of_1=True,
                              remove_feat_groups_of_2=True,
                              remove_feat_groups_of_3=True,
                              remove_feat_groups_of_4=True,
                              remove_feat_groups_of_5=True):
    # input: training data
    # output: mutually exclusive augmented version of training data, where columns are removed

    # setup
    x_all = []
    y_all = []
    x_ts_all = []
    y_ts_all = []
    x_cntry_all = []
    y_cntry_all = []

    n_feats = x.shape[2]
    n_samples = x.shape[0]

    #################################
    # loop through all samples
    for i in range(n_samples):
        print(f'samp: {i}')

        #################################
        # remove one column from train data, one at a time
        for j1 in range(n_feats):
            if remove_feat_groups_of_1:

                # get series to benchmark
                x_sample_orig = copy.deepcopy(x[i, :, :])

                # make proposal
                x_sample_mod = copy.deepcopy(x[i, :, :])

                # # zero-out feature
                x_sample_mod[:, j1] = 0.0

                # only add if this sample does not exist in the original data
                if not np.all(x_sample_orig == x_sample_mod):
                    y_sample = copy.deepcopy(y[i, :])
                    x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                    y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                    x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                    y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                    # add all at once
                    x_all.append(x_sample_mod)
                    y_all.append(y_sample)
                    x_ts_all.append(x_ts_sample)
                    y_ts_all.append(y_ts_sample)
                    x_cntry_all.append(x_cntry_sample)
                    y_cntry_all.append(y_cntry_sample)

            #################################
            # if approrpiate, start setting up for second loop
            # remove two columns from train data, one pair at a time
            exclusive_of_j1 = list(range(n_feats))
            exclusive_of_j1.remove(j1)
            for j2 in exclusive_of_j1:
                if remove_feat_groups_of_2:

                    # get series to benchmark
                    x_sample_orig = copy.deepcopy(x[i, :, :])

                    # make proposal
                    x_sample_mod = copy.deepcopy(x[i, :, :])

                    # # zero-out feature
                    x_sample_mod[:, j1] = 0.0
                    x_sample_mod[:, j2] = 0.0

                    # only add if this sample does not exist in the original data
                    if not np.all(x_sample_orig == x_sample_mod):
                        y_sample = copy.deepcopy(y[i, :])
                        x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                        y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                        x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                        y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                        # add all at once
                        x_all.append(x_sample_mod)
                        y_all.append(y_sample)
                        x_ts_all.append(x_ts_sample)
                        y_ts_all.append(y_ts_sample)
                        x_cntry_all.append(x_cntry_sample)
                        y_cntry_all.append(y_cntry_sample)

                # if approrpiate, start setting up for third loop
                # remove three columns from train data
                exclusive_of_j1j2 = list(range(n_feats))
                exclusive_of_j1j2.remove(j1)
                exclusive_of_j1j2.remove(j2)
                for j3 in exclusive_of_j1j2:
                    if remove_feat_groups_of_3:

                        # get series to benchmark
                        x_sample_orig = copy.deepcopy(x[i, :, :])

                        # make proposal
                        x_sample_mod = copy.deepcopy(x[i, :, :])

                        # # zero-out feature
                        x_sample_mod[:, j1] = 0.0
                        x_sample_mod[:, j2] = 0.0
                        x_sample_mod[:, j3] = 0.0

                        # only add if this sample does not exist in the original data
                        if not np.all(x_sample_orig == x_sample_mod):
                            y_sample = copy.deepcopy(y[i, :])
                            x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                            y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                            x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                            y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                            # add all at once
                            x_all.append(x_sample_mod)
                            y_all.append(y_sample)
                            x_ts_all.append(x_ts_sample)
                            y_ts_all.append(y_ts_sample)
                            x_cntry_all.append(x_cntry_sample)
                            y_cntry_all.append(y_cntry_sample)

                    #################################
                    # if approrpiate, start setting up for fourth loop
                    # remove four columns from train data, one pair at a time
                    exclusive_of_j1j2j3 = list(range(n_feats))
                    exclusive_of_j1j2j3.remove(j1)
                    exclusive_of_j1j2j3.remove(j2)
                    exclusive_of_j1j2j3.remove(j3)
                    for j4 in exclusive_of_j1j2j3:
                        if remove_feat_groups_of_4:

                            # get series to benchmark
                            x_sample_orig = copy.deepcopy(x[i, :, :])

                            # make proposal
                            x_sample_mod = copy.deepcopy(x[i, :, :])

                            # # zero-out feature
                            x_sample_mod[:, j1] = 0.0
                            x_sample_mod[:, j2] = 0.0
                            x_sample_mod[:, j3] = 0.0
                            x_sample_mod[:, j4] = 0.0

                            # only add if this sample does not exist in the original data
                            if not np.all(x_sample_orig == x_sample_mod):
                                y_sample = copy.deepcopy(y[i, :])
                                x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                                y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                                x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                                y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                                # add all at once
                                x_all.append(x_sample_mod)
                                y_all.append(y_sample)
                                x_ts_all.append(x_ts_sample)
                                y_ts_all.append(y_ts_sample)
                                x_cntry_all.append(x_cntry_sample)
                                y_cntry_all.append(y_cntry_sample)

                        #################################
                        # if approrpiate, start setting up for fifth loop
                        # remove five columns from train data
                        exclusive_of_j1j2j3j4 = list(range(n_feats))
                        exclusive_of_j1j2j3j4.remove(j1)
                        exclusive_of_j1j2j3j4.remove(j2)
                        exclusive_of_j1j2j3j4.remove(j3)
                        exclusive_of_j1j2j3j4.remove(j4)
                        for j5 in exclusive_of_j1j2j3j4:
                            if remove_feat_groups_of_5:

                                # get series to benchmark
                                x_sample_orig = copy.deepcopy(x[i, :, :])

                                # make proposal
                                x_sample_mod = copy.deepcopy(x[i, :, :])

                                # # zero-out feature
                                x_sample_mod[:, j1] = 0.0
                                x_sample_mod[:, j2] = 0.0
                                x_sample_mod[:, j3] = 0.0
                                x_sample_mod[:, j4] = 0.0
                                x_sample_mod[:, j5] = 0.0

                                # only add if this sample does not exist in the original data
                                if not np.all(x_sample_orig == x_sample_mod):
                                    y_sample = copy.deepcopy(y[i, :])
                                    x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                                    y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                                    x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                                    y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                                    # add all at once
                                    x_all.append(x_sample_mod)
                                    y_all.append(y_sample)
                                    x_ts_all.append(x_ts_sample)
                                    y_ts_all.append(y_ts_sample)
                                    x_cntry_all.append(x_cntry_sample)
                                    y_cntry_all.append(y_cntry_sample)

    # Once you're done iterating, stack all arrays into the correct format
    x_all = np.stack(x_all, axis=0)
    y_all = np.stack(y_all, axis=0)
    x_ts_all = np.stack(x_ts_all, axis=0)
    y_ts_all = np.stack(y_ts_all, axis=0)
    x_cntry_all = np.stack(x_cntry_all, axis=0)
    y_cntry_all = np.stack(y_cntry_all, axis=0)

    return x_all, y_all, x_ts_all, y_ts_all, x_cntry_all, y_cntry_all


def augment_by_removing_times(x,
                              y,
                              x_ts,
                              y_ts,
                              x_cntry,
                              y_cntry,
                              remove_timestep_groups_of_1=True,
                              remove_timestep_groups_of_2=True,
                              remove_timestep_groups_of_3=True,
                              remove_timestep_groups_of_4=True,
                              remove_timestep_groups_of_5=True):
    # input: training data
    # output: mutually exclusive augmented version of training data, where columns are removed

    # setup
    x_all = []
    y_all = []
    x_ts_all = []
    y_ts_all = []
    x_cntry_all = []
    y_cntry_all = []

    n_timesteps = x.shape[2]
    n_samples = x.shape[0]

    #################################
    # loop through all samples
    for i in range(n_samples):
        print(f'samp: {i}')

        #################################
        # remove one column from train data, one at a time
        for j1 in range(n_timesteps):
            if remove_timestep_groups_of_1:

                # get series to benchmark
                x_sample_orig = copy.deepcopy(x[i, :, :])

                # make proposal
                x_sample_mod = copy.deepcopy(x[i, :, :])

                # # zero-out feature
                x_sample_mod[j1, :] = 0.0

                # only add if this sample does not exist in the original data
                if not np.all(x_sample_orig == x_sample_mod):
                    y_sample = copy.deepcopy(y[i, :])
                    x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                    y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                    x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                    y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                    # add all at once
                    x_all.append(x_sample_mod)
                    y_all.append(y_sample)
                    x_ts_all.append(x_ts_sample)
                    y_ts_all.append(y_ts_sample)
                    x_cntry_all.append(x_cntry_sample)
                    y_cntry_all.append(y_cntry_sample)

            #################################
            # if approrpiate, start setting up for second loop
            # remove two columns from train data, one pair at a time
            exclusive_of_j1 = list(range(n_timesteps))
            exclusive_of_j1.remove(j1)
            for j2 in exclusive_of_j1:
                if remove_timestep_groups_of_2:

                    # get series to benchmark
                    x_sample_orig = copy.deepcopy(x[i, :, :])

                    # make proposal
                    x_sample_mod = copy.deepcopy(x[i, :, :])

                    # # zero-out feature
                    x_sample_mod[j1, :] = 0.0
                    x_sample_mod[j2, :] = 0.0

                    # only add if this sample does not exist in the original data
                    if not np.all(x_sample_orig == x_sample_mod):
                        y_sample = copy.deepcopy(y[i, :])
                        x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                        y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                        x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                        y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                        # add all at once
                        x_all.append(x_sample_mod)
                        y_all.append(y_sample)
                        x_ts_all.append(x_ts_sample)
                        y_ts_all.append(y_ts_sample)
                        x_cntry_all.append(x_cntry_sample)
                        y_cntry_all.append(y_cntry_sample)

                # if approrpiate, start setting up for third loop
                # remove three columns from train data
                exclusive_of_j1j2 = list(range(n_timesteps))
                exclusive_of_j1j2.remove(j1)
                exclusive_of_j1j2.remove(j2)
                for j3 in exclusive_of_j1j2:
                    if remove_timestep_groups_of_3:

                        # get series to benchmark
                        x_sample_orig = copy.deepcopy(x[i, :, :])

                        # make proposal
                        x_sample_mod = copy.deepcopy(x[i, :, :])

                        # # zero-out feature
                        x_sample_mod[j1, :] = 0.0
                        x_sample_mod[j2, :] = 0.0
                        x_sample_mod[j3, :] = 0.0

                        # only add if this sample does not exist in the original data
                        if not np.all(x_sample_orig == x_sample_mod):
                            y_sample = copy.deepcopy(y[i, :])
                            x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                            y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                            x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                            y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                            # add all at once
                            x_all.append(x_sample_mod)
                            y_all.append(y_sample)
                            x_ts_all.append(x_ts_sample)
                            y_ts_all.append(y_ts_sample)
                            x_cntry_all.append(x_cntry_sample)
                            y_cntry_all.append(y_cntry_sample)

                    #################################
                    # if approrpiate, start setting up for fourth loop
                    # remove four columns from train data, one pair at a time
                    exclusive_of_j1j2j3 = list(range(n_timesteps))
                    exclusive_of_j1j2j3.remove(j1)
                    exclusive_of_j1j2j3.remove(j2)
                    exclusive_of_j1j2j3.remove(j3)
                    for j4 in exclusive_of_j1j2j3:
                        if remove_timestep_groups_of_4:

                            # get series to benchmark
                            x_sample_orig = copy.deepcopy(x[i, :, :])

                            # make proposal
                            x_sample_mod = copy.deepcopy(x[i, :, :])

                            # # zero-out feature
                            x_sample_mod[j1, :] = 0.0
                            x_sample_mod[j2, :] = 0.0
                            x_sample_mod[j3, :] = 0.0
                            x_sample_mod[j4, :] = 0.0

                            # only add if this sample does not exist in the original data
                            if not np.all(x_sample_orig == x_sample_mod):
                                y_sample = copy.deepcopy(y[i, :])
                                x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                                y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                                x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                                y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                                # add all at once
                                x_all.append(x_sample_mod)
                                y_all.append(y_sample)
                                x_ts_all.append(x_ts_sample)
                                y_ts_all.append(y_ts_sample)
                                x_cntry_all.append(x_cntry_sample)
                                y_cntry_all.append(y_cntry_sample)

                        #################################
                        # if approrpiate, start setting up for fifth loop
                        # remove five columns from train data
                        exclusive_of_j1j2j3j4 = list(range(n_timesteps))
                        exclusive_of_j1j2j3j4.remove(j1)
                        exclusive_of_j1j2j3j4.remove(j2)
                        exclusive_of_j1j2j3j4.remove(j3)
                        exclusive_of_j1j2j3j4.remove(j4)
                        for j5 in exclusive_of_j1j2j3j4:
                            if remove_timestep_groups_of_5:

                                # get series to benchmark
                                x_sample_orig = copy.deepcopy(x[i, :, :])

                                # make proposal
                                x_sample_mod = copy.deepcopy(x[i, :, :])

                                # # zero-out feature
                                x_sample_mod[j1, :] = 0.0
                                x_sample_mod[j2, :] = 0.0
                                x_sample_mod[j3, :] = 0.0
                                x_sample_mod[j4, :] = 0.0
                                x_sample_mod[j5, :] = 0.0

                                # only add if this sample does not exist in the original data
                                if not np.all(x_sample_orig == x_sample_mod):
                                    y_sample = copy.deepcopy(y[i, :])
                                    x_ts_sample = copy.deepcopy(x_ts[i, :, :])
                                    y_ts_sample = copy.deepcopy(y_ts[i, :, :])
                                    x_cntry_sample = copy.deepcopy(x_cntry[i, :, :])
                                    y_cntry_sample = copy.deepcopy(y_cntry[i, :, :])

                                    # add all at once
                                    x_all.append(x_sample_mod)
                                    y_all.append(y_sample)
                                    x_ts_all.append(x_ts_sample)
                                    y_ts_all.append(y_ts_sample)
                                    x_cntry_all.append(x_cntry_sample)
                                    y_cntry_all.append(y_cntry_sample)

    # Once you're done iterating, stack all arrays into the correct format
    x_all = np.stack(x_all, axis=0)
    y_all = np.stack(y_all, axis=0)
    x_ts_all = np.stack(x_ts_all, axis=0)
    y_ts_all = np.stack(y_ts_all, axis=0)
    x_cntry_all = np.stack(x_cntry_all, axis=0)
    y_cntry_all = np.stack(y_cntry_all, axis=0)

    return x_all, y_all, x_ts_all, y_ts_all, x_cntry_all, y_cntry_all


def load_data(fold_num_test=0,
              fold_num_val=1,
              num_folds=10,
              scale_type='minmax',
              normalize_hist_cons_in_x=False,
              per_capita_flag=True,
              all_feats=None,
              feat_target=None,
              feats_complete_req=None,
              feats_nans_allowed=None,
              nans_to_zeros_after_norm=True,
              years_pre=15,
              years_post=15,
              remove_feats=True,
              remove_times=True,
              remove_feat_groups_of_1=True,
              remove_feat_groups_of_2=True,
              remove_feat_groups_of_3=False,
              remove_feat_groups_of_4=False,
              remove_feat_groups_of_5=False,
              remove_timestep_groups_of_1=True,
              remove_timestep_groups_of_2=True,
              remove_timestep_groups_of_3=False,
              remove_timestep_groups_of_4=False,
              remove_timestep_groups_of_5=False):
    # load data from compile_data, but then modify it according to the per-capita flag,
    # the number of folds, the fold number, and the normalization method.

    # get raw data and uid for raw data. This depends on what features are required and allowed.
    data_dict, df_combination_uid = compile_data([], all_feats=all_feats, feat_target=feat_target,
                                                 feats_complete_req=feats_complete_req,
                                                 feats_nans_allowed=feats_nans_allowed)

    # compile args for load_data so that we can define the
    load_data_args = {}
    load_data_args['fold_num_test'] = fold_num_test
    load_data_args['fold_num_val'] = fold_num_val
    load_data_args['num_folds'] = num_folds
    load_data_args['scale_type'] = scale_type
    load_data_args['per_capita_flag'] = per_capita_flag
    load_data_args['all_feats'] = all_feats
    load_data_args['feat_target'] = feat_target
    load_data_args['feats_complete_req'] = feats_complete_req
    load_data_args['feats_nans_allowed'] = feats_nans_allowed
    load_data_args['nans_to_zeros_after_norm'] = nans_to_zeros_after_norm
    load_data_args['years_pre'] = years_pre
    load_data_args['years_post'] = years_post
    load_data_args['remove_feat_groups_of_1'] = remove_feat_groups_of_1
    load_data_args['remove_feat_groups_of_2'] = remove_feat_groups_of_2
    load_data_args['remove_feat_groups_of_3'] = remove_feat_groups_of_3
    load_data_args['remove_feat_groups_of_4'] = remove_feat_groups_of_4
    load_data_args['remove_feat_groups_of_5'] = remove_feat_groups_of_5
    load_data_args['remove_timestep_groups_of_1'] = remove_timestep_groups_of_1
    load_data_args['remove_timestep_groups_of_2'] = remove_timestep_groups_of_2
    load_data_args['remove_timestep_groups_of_3'] = remove_timestep_groups_of_3
    load_data_args['remove_timestep_groups_of_4'] = remove_timestep_groups_of_4
    load_data_args['remove_timestep_groups_of_5'] = remove_timestep_groups_of_5

    load_data_args_string = json.dumps(load_data_args)
    load_data_args_hash = hashlib.sha256(load_data_args_string.encode('utf-8')).hexdigest()
    load_data_args_hash_folder_path = os.path.join(os.environ.get('PROJECT_ROOT'), 'data', 'processed',
                                                   df_combination_uid + f'-ypr{years_pre}-ypo{years_post}',
                                                   f'test{fold_num_test}-val{fold_num_val}-of-n{num_folds}')
    os.makedirs(load_data_args_hash_folder_path, exist_ok=True)
    load_data_args_hash_pickle_path = Path(os.path.join(load_data_args_hash_folder_path, load_data_args_hash + '.p'))

    # save to file, if not already a file
    if not load_data_args_hash_pickle_path.is_file():
        dataset = data_dict['df_all']
        dataset, dataset_norm, scaler = \
            scale_data(dataset,
                       scale_type=scale_type,
                       per_capita_flag=per_capita_flag,
                       feats_complete_req=feats_complete_req,
                       feats_nans_allowed=feats_nans_allowed,
                       nans_to_zeros_after_norm=nans_to_zeros_after_norm
                       )

        target_feat_name = get_target_feat_name(per_capita_flag=per_capita_flag)

        # return train and test data
        (train_x, train_y, val_x, val_y, test_x, test_y), \
        (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts), \
        (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry), \
        (future_x, future_x_ts, future_x_cntry), \
        (hist_x, hist_x_ts, hist_x_cntry), \
        all_cons = \
            split_dataset_cons(dataset_norm,
                               target_feat_name=target_feat_name,
                               fold_num_test=fold_num_test,
                               fold_num_val=fold_num_val,
                               num_folds=num_folds,
                               years_pre=years_pre,
                               years_post=years_post)

        if remove_feats:
            train_x_rem_feats, \
            train_y_rem_feats, \
            train_x_ts_rem_feats, \
            train_y_ts_rem_feats, \
            train_x_cntry_rem_feats, \
            train_y_cntry_rem_feats = \
                augment_by_removing_feats(train_x,
                                          train_y,
                                          train_x_ts,
                                          train_y_ts,
                                          train_x_cntry,
                                          train_y_cntry,
                                          remove_feat_groups_of_1=remove_feat_groups_of_1,
                                          remove_feat_groups_of_2=remove_feat_groups_of_2,
                                          remove_feat_groups_of_3=remove_feat_groups_of_3,
                                          remove_feat_groups_of_4=remove_feat_groups_of_4,
                                          remove_feat_groups_of_5=remove_feat_groups_of_5)

        if remove_times:
            train_x_rem_times, \
            train_y_rem_times, \
            train_x_ts_rem_times, \
            train_y_ts_rem_times, \
            train_x_cntry_rem_times, \
            train_y_cntry_rem_times = \
                augment_by_removing_times(train_x,
                                          train_y,
                                          train_x_ts,
                                          train_y_ts,
                                          train_x_cntry,
                                          train_y_cntry,
                                          remove_timestep_groups_of_1=remove_timestep_groups_of_1,
                                          remove_timestep_groups_of_2=remove_timestep_groups_of_2,
                                          remove_timestep_groups_of_3=remove_timestep_groups_of_3,
                                          remove_timestep_groups_of_4=remove_timestep_groups_of_4,
                                          remove_timestep_groups_of_5=remove_timestep_groups_of_5)

        if remove_feats:
            val_x_rem_feats, \
            val_y_rem_feats, \
            val_x_ts_rem_feats, \
            val_y_ts_rem_feats, \
            val_x_cntry_rem_feats, \
            val_y_cntry_rem_feats = \
                augment_by_removing_feats(val_x,
                                          val_y,
                                          val_x_ts,
                                          val_y_ts,
                                          val_x_cntry,
                                          val_y_cntry,
                                          remove_feat_groups_of_1=remove_feat_groups_of_1,
                                          remove_feat_groups_of_2=remove_feat_groups_of_2,
                                          remove_feat_groups_of_3=remove_feat_groups_of_3,
                                          remove_feat_groups_of_4=remove_feat_groups_of_4,
                                          remove_feat_groups_of_5=remove_feat_groups_of_5)

        if remove_times:
            val_x_rem_times, \
            val_y_rem_times, \
            val_x_ts_rem_times, \
            val_y_ts_rem_times, \
            val_x_cntry_rem_times, \
            val_y_cntry_rem_times = \
                augment_by_removing_times(val_x,
                                          val_y,
                                          val_x_ts,
                                          val_y_ts,
                                          val_x_cntry,
                                          val_y_cntry,
                                          remove_timestep_groups_of_1=remove_timestep_groups_of_1,
                                          remove_timestep_groups_of_2=remove_timestep_groups_of_2,
                                          remove_timestep_groups_of_3=remove_timestep_groups_of_3,
                                          remove_timestep_groups_of_4=remove_timestep_groups_of_4,
                                          remove_timestep_groups_of_5=remove_timestep_groups_of_5)

        if remove_feats:
            train_x = np.vstack((train_x, train_x_rem_feats))
            train_y = np.vstack((train_y, train_y_rem_feats))
            train_x_ts = np.vstack((train_x_ts, train_x_ts_rem_feats))
            train_y_ts = np.vstack((train_y_ts, train_y_ts_rem_feats))
            train_x_cntry = np.vstack((train_x_cntry, train_x_cntry_rem_feats))
            train_y_cntry = np.vstack((train_y_cntry, train_y_cntry_rem_feats))

        if remove_times:
            train_x = np.vstack((train_x, train_x_rem_times))
            train_y = np.vstack((train_y, train_y_rem_times))
            train_x_ts = np.vstack((train_x_ts, train_x_ts_rem_times))
            train_y_ts = np.vstack((train_y_ts, train_y_ts_rem_times))
            train_x_cntry = np.vstack((train_x_cntry, train_x_cntry_rem_times))
            train_y_cntry = np.vstack((train_y_cntry, train_y_cntry_rem_times))

        if remove_feats:
            val_x = np.vstack((val_x, val_x_rem_feats))
            val_y = np.vstack((val_y, val_y_rem_feats))
            val_x_ts = np.vstack((val_x_ts, val_x_ts_rem_feats))
            val_y_ts = np.vstack((val_y_ts, val_y_ts_rem_feats))
            val_x_cntry = np.vstack((val_x_cntry, val_x_cntry_rem_feats))
            val_y_cntry = np.vstack((val_y_cntry, val_y_cntry_rem_feats))

        if remove_times:
            val_x = np.vstack((val_x, val_x_rem_times))
            val_y = np.vstack((val_y, val_y_rem_times))
            val_x_ts = np.vstack((val_x_ts, val_x_ts_rem_times))
            val_y_ts = np.vstack((val_y_ts, val_y_ts_rem_times))
            val_x_cntry = np.vstack((val_x_cntry, val_x_cntry_rem_times))
            val_y_cntry = np.vstack((val_y_cntry, val_y_cntry_rem_times))

        # clean up a bit
        feat_titles = dataset_norm.columns.values.tolist()
        feat_titles.remove('country_code')
        feat_titles.remove('year_orig')

        # providing original train data by inverse transforming
        train_x_unnorm = copy.deepcopy(train_x)
        train_x_unnorm = train_x_unnorm.reshape(train_x_unnorm.shape[0] * train_x_unnorm.shape[1],
                                                train_x_unnorm.shape[2])
        train_x_unnorm[:, 1:] = scaler.inverse_transform(train_x_unnorm[:, 1:])
        train_x_unnorm = train_x_unnorm.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2])

        # providing original test data by inverse transforming
        test_x_unnorm = copy.deepcopy(test_x)
        test_x_unnorm = test_x_unnorm.reshape(test_x_unnorm.shape[0] * test_x_unnorm.shape[1], test_x_unnorm.shape[2])
        test_x_unnorm[:, 1:] = scaler.inverse_transform(test_x_unnorm[:, 1:])
        test_x_unnorm = test_x_unnorm.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2])

        # providing std data correspodning to per capita or total values, as specified
        x0_std = np.nanstd(dataset[target_feat_name].values)

        # importantly, the 0th index in our x's are NOT normalized properly!!
        # We have non-scaled values for this index in our y's and that's good,
        # but realize that we also have them leaking in our x's and also in our
        # "dataset_norm". Let's fix that!
        # get min and max values:
        x0_min = np.nanmin(dataset[target_feat_name].values)
        x0_max = np.nanmax(dataset[target_feat_name].values)

        def rescale_x(x_input, x0_min, x0_max):
            x = copy.deepcopy(x_input)
            x_zeroth = x[:, :, 0]
            x_zeroth[x_zeroth < np.finfo(np.float32).eps] = np.nan
            x_zeroth = (x_zeroth - x0_min) / (x0_max - x0_min) + 1.
            x_zeroth[np.isnan(x_zeroth)] = 0.
            x[:, :, 0] = x_zeroth
            return x

        if normalize_hist_cons_in_x:
            train_x = rescale_x(train_x, x0_min, x0_max)
            val_x = rescale_x(val_x, x0_min, x0_max)
            test_x = rescale_x(test_x, x0_min, x0_max)
            future_x = rescale_x(future_x, x0_min, x0_max)
            hist_x = rescale_x(hist_x, x0_min, x0_max)

        dataset_norm_feat_values = dataset_norm[target_feat_name].values
        dataset_norm_feat_values[dataset_norm_feat_values < np.finfo(np.float32).eps] = np.nan
        dataset_norm_feat_values = (dataset_norm_feat_values - x0_min) / (x0_max - x0_min) + 1.
        dataset_norm_feat_values[np.isnan(dataset_norm_feat_values)] = 0.
        dataset_norm[target_feat_name] = dataset_norm_feat_values

        print(f'saving to pickle: {load_data_args_hash_pickle_path}')

        # saving dict of dfs
        pickle.dump(
            (dataset, dataset_norm,
             (train_x, train_y, val_x, val_y, test_x, test_y),
             (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts),
             (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry),
             (future_x, future_x_ts, future_x_cntry),
             (hist_x, hist_x_ts, hist_x_cntry),
             (feat_titles, train_x_unnorm, test_x_unnorm, x0_std, all_cons)
             ), open(load_data_args_hash_pickle_path, "wb"), protocol=4)

    else:

        print(f'loading from pickle: {load_data_args_hash_pickle_path}')

        (dataset, dataset_norm,
         (train_x, train_y, val_x, val_y, test_x, test_y),
         (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts),
         (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry),
         (future_x, future_x_ts, future_x_cntry),
         (hist_x, hist_x_ts, hist_x_cntry),
         (feat_titles, train_x_unnorm, test_x_unnorm, x0_std, all_cons)
         ) = \
            pickle.load(open(load_data_args_hash_pickle_path, "rb"))

    return (dataset, dataset_norm,
            (train_x, train_y, val_x, val_y, test_x, test_y),
            (train_x_ts, train_y_ts, val_x_ts, val_y_ts, test_x_ts, test_y_ts),
            (train_x_cntry, train_y_cntry, val_x_cntry, val_y_cntry, test_x_cntry, test_y_cntry),
            (future_x, future_x_ts, future_x_cntry),
            (hist_x, hist_x_ts, hist_x_cntry),
            (feat_titles, train_x_unnorm, test_x_unnorm, x0_std, all_cons)
            )


def main(args_raw):
    # data_dict = compile_data(args_raw)

    args = parse_args(args_raw)
    args = args.__dict__

    import time

    start = time.time()
    load_data(fold_num_test=args['fold_num_test'],
              fold_num_val=args['fold_num_val'],
              num_folds=args['num_folds'],
              scale_type=args["scale_type"],
              per_capita_flag=args['per_capita_flag'],
              all_feats=args['all_feats'],
              feat_target=args['feat_target'],
              feats_complete_req=args['feats_complete_req'],
              feats_nans_allowed=args['feats_nans_allowed'],
              nans_to_zeros_after_norm=args['nans_to_zeros_after_norm'],
              years_pre=args['years_pre'],
              years_post=args['years_post'],
              remove_feat_groups_of_1=args['remove_feat_groups_of_1'],
              remove_feat_groups_of_2=args['remove_feat_groups_of_2'],
              remove_feat_groups_of_3=args['remove_feat_groups_of_3'],
              remove_feat_groups_of_4=args['remove_feat_groups_of_4'],
              remove_feat_groups_of_5=args['remove_feat_groups_of_5'],
              remove_timestep_groups_of_1=args['remove_timestep_groups_of_1'],
              remove_timestep_groups_of_2=args['remove_timestep_groups_of_2'],
              remove_timestep_groups_of_3=args['remove_timestep_groups_of_3'],
              remove_timestep_groups_of_4=args['remove_timestep_groups_of_4'],
              remove_timestep_groups_of_5=args['remove_timestep_groups_of_5'])

    end = time.time()
    print(f'time: {end - start}')

    print(
        f'done generating data splits for fold_num_test: {args["fold_num_test"]} / fold_num_val: {args["fold_num_val"]} out of num_folds: {args["num_folds"]}')


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
