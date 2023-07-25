import logging
import os
import db_dtypes
import pandas as pd 
import pandera as pa
from google.cloud import bigquery
from Deepl_reproduction.ETL.extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.logs.logs import main

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

def load_data(data, project_id="deepl-reproduction", dataset_id="raw_data", table_name="raw_wikipedia", client=bigquery.Client()) -> None:
    table_ref=client.dataset(dataset_id).table(table_name)
    client.insert_rows_json(table_ref, data)
    logging.info(f"Data were successfully pushed in dataset {dataset_id} in table {table_name}")


if __name__=="__main__":
    page, content=get_wikipedia_article()
    data=[{"page_name":page, "content":content}]
    df_wikipedia=pd.DataFrame(data)
    schema_wikipedia_raw.validate(df_wikipedia)

    if page not in unique_page:
        load_data(data)
    else:
        logging.info(f"The page {page} was already in the database and was skipped")