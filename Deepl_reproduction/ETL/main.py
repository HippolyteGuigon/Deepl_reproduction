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
from typing import List 
from google.cloud import bigquery
from flask import Flask, request


from Deepl_reproduction.configs.confs import load_conf, clean_params
from Deepl_reproduction.ETL.extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.ETL.transform.transform_wikipedia import treat_article, translate_content
from Deepl_reproduction.ETL.load.load_wikipedia import indexing, load_raw_data, load_processed_data

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main_params=load_conf("configs/main.yml", include=True)
deepl_api_key=main_params["deepl"]["API_KEY"]

@functions_framework.http
def wikipedia_etl_pipeline(request):
    sql_query = '''
    SELECT
        *
    FROM
        `deepl-reprodution.raw_data.raw_wikipedia`
    '''

    schema_wikipedia_cleaned = pa.DataFrameSchema(
        {   
            "page_name": pa.Column(pa.String, nullable=False),
            "page_name_id": pa.Column(pa.Int, nullable=False),
            "content": pa.Column(pa.String, nullable=False),
            "content_translated" : pa.Column(pa.String, nullable=False),
        }
    )

    schema_wikipedia_raw = pa.DataFrameSchema(
        {   
            "page_name": pa.Column(pa.String, nullable=False),
            "content": pa.Column(pa.String, nullable=False),
        }
    )

    query_job = client.query(sql_query)
    all_dataframe=query_job.to_dataframe()
    unique_page = all_dataframe.page_name.to_list()

    schema_wikipedia_raw.validate(all_dataframe)
    logging.info("Schema was validated for the all dataframe")

    for _ in range(5):
        page, content=get_wikipedia_article()
        data=[{"page_name":page, "content":content}]
        df_wikipedia=pd.DataFrame(data)
        schema_wikipedia_raw.validate(df_wikipedia)
        df_wikipedia["content"]=df_wikipedia["content"].apply(lambda article: treat_article(article))
        df_wikipedia=indexing(df_wikipedia)
        df_wikipedia=translate_content(df_wikipedia)
        schema_wikipedia_cleaned.validate(df_wikipedia)    

        if page not in unique_page:
            load_raw_data(data)
            for _, row in df_wikipedia.iterrows():
                load_processed_data([{"page_name":row["page_name"], "page_name_id":row["page_name_id"],"content":row["content"], "content_translated":row["content_translated"]}])
        else:
            logging.info(f"The page {page} was already in the database and was skipped")

    return "Success"

if __name__=="__main__":
    wikipedia_etl(request)