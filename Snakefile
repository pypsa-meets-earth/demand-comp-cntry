# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from os.path import normpath, exists, isdir
from shutil import copyfile

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

if not exists("config.yaml"):
    copyfile("config.default.yaml", "config.yaml")

if not exists("out"):
    os.mkdir(os.path.join(os.getcwd(), "out"))

if not exists("cache"):
    os.mkdir(os.path.join(os.getcwd(), "cache"))


configfile: "config.yaml"


wildcard_constraints:
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    ll="(v|c)([0-9\.]+|opt|all)|all",
    opts="[-+a-zA-Z0-9\.]*",


# rule run_all:
#     input: expand("results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc", **config['scenario'])

rule download_data:
    output:
        cccodes="data/raw/country-and-continent-codes-list-csv_csv.zip",
        borders="data/raw/TM_WORLD_BORDERS-0.3.zip",
        wbpop="data/raw/wb_pop.zip",
        wbgdp_per_capita="data/raw/GDPpc-wb-1960-2018.zip",
        wbgdp="data/raw/GDP-wb-1960-2019.zip",
        ungdp="data/raw/UN_GDP_region_2015dollars.zip",
        ungdp_somalia="data/raw/UNdata_Export_20201006_214431107_SomaliaGDPpc.zip",
        wbrain="data/raw/rainfall_1960-2016.zip",
        wbtemperature="data/raw/temperature_1960-2016.zip",
        battle_deaths="data/raw/API_VC.BTL.DETH_DS2_en_csv_v2_2167203.zip",
        iea_cdd="data/raw/CDD_18_IEA_20210406.zip",
        iea_hdd="data/raw/HDD_18_IEA_20210406.zip",
        atalla_cdd_hdd="data/raw/CDD_HDD_18_Atalla_20210406.zip"
    log: "logs/download.log"
    script: "demand/data/download_data.py"


rule run_arima:
    input:
        cccodes="data/raw/country-and-continent-codes-list-csv_csv.zip"
    log: "logs/download.log"
    script: "demand/models/run_arima.py"
