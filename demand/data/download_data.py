import os, sys, time, zipfile
import urllib.request

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def download_raw_data(source_url, target_path, data_label):
    try:
        if not os.path.exists(target_path):
            urllib.request.urlretrieve(source_url, target_path)
            print('successfully downloaded ', data_label)

            with zipfile.ZipFile(target_path, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(target_path))
            print('successfully extracted ', data_label)

    except:
        time.sleep(60)
        print('The download was interrupted. Trying again.')
        download_raw_data(source_url, target_path, data_label)

# raw
if not os.path.exists(os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw')):
    os.makedirs(os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw'))


# Country and continent codes list
source_url = "https://www.dropbox.com/s/01hg0ysjt7sj0zz/country-and-continent-codes-list-csv_csv.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/country-and-continent-codes-list-csv_csv.zip')
data_label = "Country and continent codes list"
download_raw_data(source_url, target_path, data_label)

# World country borders shapefile
source_url = "https://www.dropbox.com/s/tflzjyq6noz5veb/TM_WORLD_BORDERS-0.3.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/TM_WORLD_BORDERS-0.3.zip')
data_label = "World country borders shapefile"
download_raw_data(source_url, target_path, data_label)

# WB Population Data
source_url = "https://www.dropbox.com/s/22jszjd1sohk6g7/wb_pop.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/wb_pop.zip')
data_label = "WB Population Data"
download_raw_data(source_url, target_path, data_label)

# WB GDP per capita Data
source_url = "https://www.dropbox.com/s/40gkpaif2oz5xph/GDPpc-wb-1960-2018.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/GDPpc-wb-1960-2018.zip')
data_label = "WB GDP pc Data"
download_raw_data(source_url, target_path, data_label)

# WB GDP Data
source_url = "https://www.dropbox.com/s/6t37n56jzpdpp20/GDP-wb-1960-2019.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/GDP-wb-1960-2019.zip')
data_label = "WB GDP Data 2010 dollars"
download_raw_data(source_url, target_path, data_label)

# UN GDP Region Data
source_url = "https://www.dropbox.com/s/e3ywyyraxk0jtgs/UN_GDP_region_2015dollars.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/UN_GDP_region_2015dollars.zip')
data_label = "UN GDP Region Data 2015 dollars"
download_raw_data(source_url, target_path, data_label)

# UN Somalia GDP Data
source_url = "https://www.dropbox.com/s/r7w07j1083escof/UNdata_Export_20201006_214431107_SomaliaGDPpc.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/UNdata_Export_20201006_214431107_SomaliaGDPpc.zip')
data_label = "UN Somalia GDP Data"
download_raw_data(source_url, target_path, data_label)

# WB Rainfall Data
source_url = "https://www.dropbox.com/s/500p8cthhsjlnz6/rainfall_1960-2016.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/rainfall_1960-2016.zip')
data_label = "WB Rainfall Data"
download_raw_data(source_url, target_path, data_label)

# WB Temperature Data
source_url = "https://www.dropbox.com/s/d7jfz4laab51g4h/temperature_1960-2016.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/temperature_1960-2016.zip')
data_label = "WB Temperature Data"
download_raw_data(source_url, target_path, data_label)

# WB Battle-Related Deaths Data
source_url = "https://www.dropbox.com/s/5gsczk7ntec7k5k/API_VC.BTL.DETH_DS2_en_csv_v2_2167203.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/API_VC.BTL.DETH_DS2_en_csv_v2_2167203.zip')
data_label = "WB Battle-Related Deaths Data"
download_raw_data(source_url, target_path, data_label)

# IEA CDD
source_url = "https://www.dropbox.com/s/ckum91no8z71t7r/CDD_18_IEA_20210406.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/CDD_18_IEA_20210406.zip')
data_label = "IEA CDD"
download_raw_data(source_url, target_path, data_label)

# IEA HDD
source_url = "https://www.dropbox.com/s/np09gni0911ais6/HDD_18_IEA_20210406.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/HDD_18_IEA_20210406.zip')
data_label = "IEA HDD"
download_raw_data(source_url, target_path, data_label)

# Atalla CDD/HDD
source_url = "https://www.dropbox.com/s/lbj680tg6x7f4jv/CDD_HDD_18_Atalla_20210406.zip?dl=1"
target_path = os.path.join(os.environ.get("PROJECT_ROOT"), r'data/raw/CDD_HDD_18_Atalla_20210406.zip')
data_label = "Atalla CDD/HDD"
download_raw_data(source_url, target_path, data_label)
