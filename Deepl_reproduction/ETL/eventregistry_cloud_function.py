import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
import requests
import re 
import pandas as pd
import wikipedia 
import deepl
import functions_framework
import importlib    
import sys
from typing import List 
from google.cloud import bigquery
from flask import Flask, request

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

from Deepl_reproduction.configs.confs import load_conf, clean_params
from Deepl_reproduction.ETL.extract.event_registry_source import get_eventregistry_article
from Deepl_reproduction.ETL.transform.transform_eventregistry import text_cleaning, translate_content
from Deepl_reproduction.ETL.load.load_eventregistry import load_raw_data_eventregistry, load_processed_data_eventregistry

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

main_params=load_conf("configs/main.yml", include=True)
deepl_api_key=main_params["deepl"]["API_KEY"]

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

sql_query = '''
SELECT
    *
FROM
    `deepl-reprodution.raw_data.raw_eventregistry`
'''

@functions_framework.http
def eventregistry_etl_pipeline(request):

    query_job = client.query(sql_query)
    all_dataframe=query_job.to_dataframe()
    unique_uri = all_dataframe.uri.to_list()

    schema_eventregistry_raw.validate(all_dataframe)
    logging.info("Schema was validated for the all dataframe")

    data=get_eventregistry_article(max_limit=1)
    data_cleaned=text_cleaning(data)    
    df_eventregistry=pd.DataFrame.from_dict(data_cleaned,orient='index').transpose()
    df_eventregistry["uri"]=df_eventregistry["uri"].astype(int)
    schema_eventregistry_raw.validate(df_eventregistry)
    df_eventregistry=translate_content(df_eventregistry)
    schema_eventregistry_processed.validate(df_eventregistry) 
    data_cleaned=df_eventregistry.to_dict("list")
    
    for index, uri in enumerate(data["uri"]):
        if data["uri"] in unique_uri:
            logging.info(f"The uri {data['uri']} was already in the database and was skipped")
            continue
        load_raw_data_eventregistry([{"uri":int(data["uri"][0]), "title":data["title"][0], "text":data["text"][0]}])
        logging.info("Raw Eventregistry data succesfully pushed in the dataframe")
        for _, row in df_eventregistry.iterrows():
                load_processed_data_eventregistry([{"uri":int(row["uri"]), "title":str(row["title"]), "title_processed":str(row["title_processed"]),"text":str(row["text"]), "text_processed":str(row["text_processed"])}])

    return "Success"