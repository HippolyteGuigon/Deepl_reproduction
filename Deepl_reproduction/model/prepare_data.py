import os
import pandas as pd 
import sys 
from utils import *
sys.path.insert(0,os.path.join(os.getcwd(),"Deepl_reproduction/pipeline"))
from data_loading import *
from Deepl_reproduction.logs.logs import main

main()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

data_path="Deepl_reproduction/model/data/full_df.csv"

if not os.path.exists(data_path):
    if not os.path.exists("Deepl_reproduction/model/data"):
        os.mkdir("Deepl_reproduction/model/data")
    data=load_all_data(language="ja")
    data.to_csv(data_path,index=False)

prepare_data(data_folder=data_path,
             euro_parl=True,
             common_crawl=True,
             news_commentary=True,
             min_length=3,
             max_length=150,
             max_length_ratio=2.,
             retain_case=True
             )
