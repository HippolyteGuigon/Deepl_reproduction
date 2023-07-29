import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from Deepl_reproduction.ETL.transform.transform_wikipedia import treat_article
from Deepl_reproduction.ETL.extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.logs.logs import main

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

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




if __name__=="__main__":
    page, content=get_wikipedia_article()
    data=[{"page_name":page, "content":content}]
    df_wikipedia=pd.DataFrame(data)
    schema_wikipedia_raw.validate(df_wikipedia)
    df_wikipedia["content"]=df_wikipedia["content"].apply(lambda article: treat_article(article))
    df_wikipedia=indexing(df_wikipedia)
    schema_wikipedia_cleaned.validate(df_wikipedia)    

    if page not in unique_page:
        load_raw_data(data)
        for _, row in df_wikipedia.iterrows():
            load_processed_data([{"page_name":row["page_name"], "page_name_id":row["page_name_id"],"content":row["content"]}])
    else:
        logging.info(f"The page {page} was already in the database and was skipped")