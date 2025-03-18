# Import
from kaggle.api.kaggle_api_extended import KaggleApi

# get dataset directly from kaggle
api = KaggleApi()
api.authenticate()

dataset_name = "chitwanmanchanda/fraudulent-transactions-data"  
dataset_folder = "C:/Users/Winni/.kaggle"

api.dataset_download_files(dataset_name,  
                           path= dataset_folder,
                           unzip=True)

