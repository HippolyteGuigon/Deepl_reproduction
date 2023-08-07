import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from Deepl_reproduction.logs.logs import main

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

if "workspace" not in os.getcwd():
    main()

schema_eventregistry_processed = pa.DataFrameSchema(
        {   
            "uri": pa.Column(pa.Int, nullable=False),
            "title": pa.Column(pa.String, nullable=False),
            "title_processed": pa.Column(pa.String, nullable=False),
            "text" : pa.Column(pa.String, nullable=False),
            "text_processed" : pa.Column(pa.String, nullable=False),
        }
    )

schema_eventregistry_raw = pa.DataFrameSchema(
    {   
        "uri": pa.Column(pa.Int, nullable=False),
        "title": pa.Column(pa.String, nullable=False),
        "text": pa.Column(pa.String, nullable=False),
    }
)

def load_raw_data_eventregistry(data, project_id="deepl-reproduction", dataset_id="raw_data", table_name="raw_eventregistry", client=bigquery.Client()) -> None:
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

def load_processed_data_eventregistry(data, project_id="deepl-reproduction", dataset_id="processed_data", table_name="processed_eventregistry", client=bigquery.Client()) -> None:
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