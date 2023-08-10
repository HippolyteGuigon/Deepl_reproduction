import os 
import pandas as pd
from google.cloud import bigquery

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

all_tables_id_query='''
SELECT table_id
FROM `deepl-reprodution.processed_data.__TABLES__`
'''

client = bigquery.Client()

query_job = client.query(all_tables_id_query)

results = query_job.result()

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