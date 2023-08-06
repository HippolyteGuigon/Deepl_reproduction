import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from Deepl_reproduction.ETL.transform.transform_wikipedia import treat_article, translate_content
from Deepl_reproduction.ETL.extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.logs.logs import main

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

if "workspace" not in os.getcwd():
    main()

def load_raw_data(data, project_id="deepl-reproduction", dataset_id="raw_data", table_name="raw_eventregistry", client=bigquery.Client()) -> None:
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

def load_processed_data(data, project_id="deepl-reproduction", dataset_id="processed_data", table_name="processed_eventregistry", client=bigquery.Client()) -> None:
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