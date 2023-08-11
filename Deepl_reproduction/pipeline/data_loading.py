import os 
import pandas as pd
import mysql.connector
import pymysql
import logging
from sqlalchemy import create_engine
from google.cloud import bigquery
from Deepl_reproduction.configs.confs import load_conf, clean_params

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

engine = create_engine("mysql+mysqldb://scott:tiger@localhost/foo")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

all_tables_id_query='''
SELECT table_id
FROM `deepl-reprodution.processed_data.__TABLES__`
'''

client = bigquery.Client()

query_job = client.query(all_tables_id_query)

results = query_job.result()

main_params=load_conf("configs/main.yml", include=True)
main_params=clean_params(main_params)

db_user = main_params["db_user"]
db_password = main_params["db_password"]
db_host = main_params["db_host"]
db_name = main_params["db_name"]

def get_dataframe_from_bq(table_id: str, project_id: str="deepl-reprodution", dataset_id: str="processed_data")->pd.DataFrame:
    """
    The goal of this function
    is to get all the data available
    in a bigquery table in a DataFrame
    
    Arguments:
        -table_id: str: The name of the
        BigQuery table to be loaded
        -project_id: str: The project 
        id containing the data
        -dataset_id: str: The dataset id
        containing the data
    Returns:
        -loaded_df: pd.DataFrame: The 
        DataFrame containing the full data
    """

    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    """

    client = bigquery.Client()
    query_job = client.query(query)
    loaded_df=query_job.to_dataframe()

    return loaded_df

def load_all_data()->pd.DataFrame:
    """
    The goal of this function is
    to load all tables in the bigquery
    to have them ready for the Machine 
    Learning pipeline
    
    Arguments:
        -None
    Returns:
        -full_data: pd.DataFrame: The DataFrame
        containing all data
    """

    full_data=pd.DataFrame(columns=["french", "english"])

    client = bigquery.Client()
    query_job = client.query(all_tables_id_query)

    results = query_job.result()

    table_id = [row.table_id for row in results]

    for id in table_id:
        if id=="processed_eventregistry":
            df=get_dataframe_from_bq(id)
            full_data=pd.concat([full_data,df[["title_processed", "title"]].rename(columns={"title_processed":"french", "title":"english"})])
            full_data=pd.concat([full_data,df[["text_processed", "text"]].rename(columns={"text_processed":"french", "text":"english"})])
        elif id=="processed_wikipedia":
            df=get_dataframe_from_bq(id)
            full_data=pd.concat([full_data,df[["content", "content_translated"]].rename(columns={"content":"french", "content_translated":"english"})])
    full_data.drop_duplicates(inplace=True)

    return full_data

def load_data_to_front_database()->None:
    full_data=load_all_data()
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')
    full_data.to_sql('deepl_table', con=engine, if_exists='replace', index=False)

    logging.info("Data successfuly pushed to the front database")

def load_data()->None:
    engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}')
    query = "SELECT * FROM deepl_table"
    data = pd.read_sql_query(query, engine)

    return data 

if __name__ == '__main__':
    load_data_to_front_database()
    print(load_data())