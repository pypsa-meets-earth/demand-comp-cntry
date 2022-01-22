# demand-comp-cntry
A competition for forecasting electricity demand at the country-level using a standard backtesting framework

## Introduction
This repo makes scripts available for downloading and compiling country-level data to be used in electricity demand forecasting at the country level. The goal of this repo is to encourage collaborative and competitive efforts towards the use of machine learning backtesting frameworks for forecasting electricity consumption and to use trained models to predict future consumption at the country-level. The results of such models can be used directly as input to the [pypsa-africa](https://github.com/pypsa-meets-africa/pypsa-africa) repository and other similar modeling efforts. 

## Technical Background
Before getting started, it is recommended that users of and contributors to this repository have some background on backtesting, cross-validation, and probabilistic forecasting. Here are a few links to get started:
* backtesting: https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/
* cross-validation: https://arxiv.org/abs/1811.12808
* probabilistic forecasting for demand: Lee et al. working paper,  https://www.energyeconomicgrowth.org/sites/default/files/2021-09/Lee_et_al-How_probabilistic_electricity_demand_forecasts.pdf


## Setup
* Set path to project root:
  ~~~~
  cd <PATH>/demand-comp-cntry
  ~~~~

* Install conda environment:
  ~~~~
  conda env create -f envs/environment.yml
  ~~~~

* Activate environment:
  ~~~~
  conda activate demand-comp
  ~~~~

  ~~~~
* Download data:
  ~~~~
  snakemake --cores 1 download_data
  ~~~~

* Ensure you have [World energy statistics (Edition 2020)](https://www.oecd-ilibrary.org/energy/data/iea-world-energy-statistics-and-balances_enestats-data-en#archive) data, filename `iea_wes_2020-68578195-en.zip`, in directory `<PROJECT_ROOT>/data/raw/iea_wes_2020-68578195-en.zip` (currently manual download)
* Ensure you have [World energy balances (Edition 2020)](https://www.oecd-ilibrary.org/energy/data/iea-world-energy-statistics-and-balances_enestats-data-en#archive) data, filename `iea_web_2020-cde01922-en.zip`, in directory `<PROJECT_ROOT>/data/raw/iea_web_2020-cde01922-en.zip` and unzip these (currently manual download)

* Run test script using ARIMA models:
  ~~~~
  snakemake --cores 1 run_arima
  ~~~~

* Analyse the outputs in demand-comp-cntry/out 