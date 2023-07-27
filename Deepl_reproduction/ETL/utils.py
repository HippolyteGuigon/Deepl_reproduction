from google.cloud import bigquery
import os 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

def create_wikipedia_table(client=client)->None:
    sql_query_raw='''
    CREATE TABLE raw_data.raw_wikipedia
    (
    page_name STRING,
    content STRING
    );
    '''

    sql_query_processed='''
    CREATE TABLE processed_data.processed_wikipedia
    (
    page_name STRING,
    page_name_id INT64,
    content STRING
    );
    '''

    query_job = client.query(sql_query_raw)
    query_job = client.query(sql_query_processed)

if __name__=="__main__":
    create_wikipedia_table()