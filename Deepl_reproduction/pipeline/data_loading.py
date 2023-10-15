import os
import pandas as pd
import logging
import concurrent.futures
from sqlalchemy import create_engine
from google.cloud import bigquery

from Deepl_reproduction.configs.confs import load_conf, clean_params

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

kaggle_length = main_params["english_dataset_size"]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

engine = create_engine("mysql+mysqldb://scott:tiger@localhost/foo")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deepl_api_key.json"

all_tables_id_query = """
SELECT table_id
FROM `deepl-reprodution-401020.processed_data.__TABLES__`
"""


client = bigquery.Client()

query_job = client.query(all_tables_id_query)

results = query_job.result()

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

db_user = main_params["db_user"]
db_password = main_params["db_password"]
db_host = main_params["db_host"]
db_name = main_params["db_name"]


def get_dataframe_from_bq(
    table_id: str,
    project_id: str = "deepl-reprodution-401020",
    dataset_id: str = "processed_data",
    kaggle_length: int = kaggle_length,
) -> pd.DataFrame:
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
        -kaggle_length: str: How many data
        should be loaded from the Kaggle dataset
        of
    Returns:
        -loaded_df: pd.DataFrame: The
        DataFrame containing the full data
    """

    if table_id == "processed_kaggle_dataset":
        query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE LENGTH(english)<1000 AND LENGTH(french)<1000
        LIMIT {kaggle_length}
        """

    else:
        query = f"""
        SELECT *
        FROM `{project_id}.{dataset_id}.{table_id}`
        """

    client = bigquery.Client()
    query_job = client.query(query)
    loaded_df = query_job.to_dataframe()

    return loaded_df

def parallel_get_dataframe_from_bq(table_ids, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda table_id: get_dataframe_from_bq(table_id, **kwargs), table_ids))
    return results

def load_all_data(
    kaggle_length: int = kaggle_length, language: str = "en"
) -> pd.DataFrame:
    """
    The goal of this function is
    to load all tables in the bigquery
    to have them ready for the Machine
    Learning pipeline

    Arguments:
        -kaggle_length: str: The
        number of sentences that
        should be extracted from
        the Kaggle dataset
    Returns:
        -full_data: pd.DataFrame: The DataFrame
        containing all data
    """

    assert language in [
        "en",
        "ja",
    ], "The language tables to load in the front database must be en or ja"

    logging.info(
        f"Loading all data on SQL front database with kaggle\
              size: {kaggle_length}, and language {language}"
    )

    if language == "en":
        full_data = pd.DataFrame(columns=["french", "english"])

        client = bigquery.Client()
        query_job = client.query(all_tables_id_query)

        results = query_job.result()

        table_id = [row.table_id for row in results]

        for id in table_id:
            
            if id == "processed_eventregistry":
                logging.info(f"Loading the {id} table")
                df = get_dataframe_from_bq(id)
                full_data = pd.concat(
                    [
                        full_data,
                        df[["title_processed", "title"]].rename(
                            columns={"title_processed": "french", "title": "english"}
                        ),
                    ]
                )
                full_data = pd.concat(
                    [
                        full_data,
                        df[["text_processed", "text"]].rename(
                            columns={"text_processed": "french", "text": "english"}
                        ),
                    ]
                )
            elif id == "processed_wikipedia":
                logging.info(f"Loading the {id} table")
                df = get_dataframe_from_bq(id)
                full_data = pd.concat(
                    [
                        full_data,
                        df[["content", "content_translated"]].rename(
                            columns={
                                "content": "french",
                                "content_translated": "english",
                            }
                        ),
                    ]
                )
            elif id == "processed_kaggle_dataset":
                logging.info(f"Loading the {id} table")
                df=parallel_get_dataframe_from_bq(id)
                #df = get_dataframe_from_bq(id, kaggle_length=kaggle_length)
                full_data = pd.concat([full_data, df[["french", "english"]]])
            else:
                continue

    elif language == "ja":
        full_data = pd.DataFrame(columns=["french", "japanese"])

        client = bigquery.Client()
        query_job = client.query(all_tables_id_query)

        results = query_job.result()

        table_id = [row.table_id for row in results]

        for id in table_id:
            
            if id=="processed_japanese_opus":
                logging.info(f"Loading the {id} table")
                df = get_dataframe_from_bq(id)
                full_data = pd.concat([full_data, df[["french", "japanese"]]])
            else:
                continue

    full_data.drop_duplicates(inplace=True)

    return full_data


def load_data_to_front_database(
    kaggle_length: int = kaggle_length, language: str = "en"
) -> None:
    full_data = load_all_data(kaggle_length=kaggle_length, language=language)
    engine = create_engine(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    )
    full_data.to_sql("deepl_table", con=engine, if_exists="replace", index=False)

    logging.info(
        f"Data successfuly pushed to the front database and is of size: {full_data.shape[0]}"
    )


def load_data() -> None:
    engine = create_engine(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    )
    query = "SELECT * FROM deepl_table"
    data = pd.read_sql_query(query, engine)

    return data
