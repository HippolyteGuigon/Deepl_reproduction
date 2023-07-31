import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from ..transform.transform_wikipedia import treat_article, translate_content
from ..extract.wikipedia_source import get_wikipedia_article
from ...logs.logs import main

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

def indexing(data: pd.DataFrame)->pd.DataFrame:
    """
    The goal of this funcction is, 
    once the data are loaded, to 
    index the DataFrame accordingly 
    and have one row per sentence
    
    Arguments:
        -data: pd.DataFrame: The 
        DataFrame with wikipedia data
    Returns:
        -indexed_dataframe: pd.DataFrame:
        The indexed dataframe
    """

    data["page_name_id"]=data["content"].apply(lambda content_list: list(range(1,len(content_list)+1)))
    indexed_dataframe=data.explode(["page_name_id","content"])
    indexed_dataframe=indexed_dataframe[["page_name","page_name_id", "content"]]
    indexed_dataframe["page_name_id"]=indexed_dataframe["page_name_id"].astype(int)

    return indexed_dataframe

def load_raw_data(data, project_id="deepl-reproduction", dataset_id="raw_data", table_name="raw_wikipedia", client=bigquery.Client()) -> None:
    """
    The goal of this function is 
    to load raw data (before they 
    were processed) in the database
    
    Arguments:
        -data: pd.DataFrame: The raw data 
    Returns:
        -None
    """
    table_ref=client.dataset(dataset_id).table(table_name)
    client.insert_rows_json(table_ref, data)
    logging.info(f"Raw data were successfully pushed in dataset {dataset_id} in table {table_name}")

def load_processed_data(data, project_id="deepl-reproduction", dataset_id="processed_data", table_name="processed_wikipedia", client=bigquery.Client()) -> None:
    """
    The goal of this function is 
    to load raw data (before they 
    were processed) in the database
    
    Arguments:
        -data: pd.DataFrame: The raw data 
    Returns:
        -None
    """
    table_ref=client.dataset(dataset_id).table(table_name)
    client.insert_rows_json(table_ref, data)
    logging.info(f"Processed data were successfully pushed in dataset {dataset_id} in table {table_name}")