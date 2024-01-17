import os
from zipfile import ZipFile
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
def main(argv):
        # Set your Kaggle API key
    api_key = argv[1]
    os.environ["KAGGLE_USERNAME"] = argv[2]
    os.environ["KAGGLE_KEY"] = api_key

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
    main(sys.argv)
