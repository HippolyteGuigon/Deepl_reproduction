import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from load.load_wikipedia import indexing, load_raw_data, load_processed_data
from transform.transform_wikipedia import treat_article, translate_content
from extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.logs.logs import main

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

logging.warning(f"You are here: {os.getcwd()}")

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

def wikipedia_etl():
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