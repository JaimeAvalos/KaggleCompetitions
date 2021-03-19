
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

import config

api = KaggleApi()
api.authenticate()

api.competition_download_files(config.KAGGLE_NAME,  path='../input')

with zipfile.ZipFile("../input/" + config.KAGGLE_NAME + ".zip","r") as zf:
    zf.extractall(path = "../input/")


#kaggle competitions download -c KAGGLE_NAME -p ../input