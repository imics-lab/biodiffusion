import getpass
import os
from zipfile import ZipFile
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
def main():
        # Set your Kaggle API key
    load_dotenv()
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    if kaggle_username == '':
        kaggle_username = getpass.getpass('Enter your Kaggle username: ')
        os.environ.__setitem__('KAGGLE_USERNAME', kaggle_username)
    if kaggle_key == '':
        kaggle_key = getpass.getpass('Enter your Kaggle API key: ')
        os.environ.__setitem__('KAGGLE_KEY', kaggle_key)

    # Set dataset and destination paths
    dataset_name = "shayanfazeli/heartbeat"
    download_path = "./"
    zip_file_path = download_path + "heartbeat.zip"
    extracted_folder_path = download_path + "heartbeat"

    # Download dataset
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)

    # Extract downloaded zip file
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)

    print("Dataset downloaded and extracted successfully.")


if __name__=="__main__":
    main()
