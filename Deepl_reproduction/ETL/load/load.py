import logging
import os
from google.cloud import bigquery
from Deepl_reproduction.ETL.extract.wikipedia_source import get_wikipedia_article
from Deepl_reproduction.logs.logs import main

project="deepl-reproduction"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deepl_api_key.json"

def load_data(data, project_id="deepl-reproduction", dataset_id="raw_data", table_name="raw_wikipedia", client=bigquery.Client()) -> None:
    table_ref=client.dataset(dataset_id).table(table_name)
    client.insert_rows_json(table_ref, data)
    logging.info(f"Data were successfully pushed in dataset {dataset_id} in table {table_name}")


if __name__=="__main__":
    page, content=get_wikipedia_article()
    data=[{"page_name":page, "content":content}]
    load_data(data)