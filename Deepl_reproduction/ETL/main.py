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
from typing import List 
from google.cloud import bigquery
from flask import Flask, request
import functions_framework

app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="deepl_api_key.json"

client = bigquery.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


API_KEY="59091169-a8eb-9609-b264-4cebdbf06970"

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

def get_wikipedia_article(language:str="fr", random=True, **kwargs)->List[str]:
    """
    The goal of this function is to 
    retrieve a all wikipedia article 
    given a keyword
    
    Arguemnts:
        -keyword: str: The keyword used
        for searching a wikipedia article
        -random: bool: Whether or not the 
        wikipedia article should be chosen
        randomly
        -**kwargs: dict: Optional argument
    Returns:
        -summary: str: The summary of the article
        -article: str: The all article
    """

    wikipedia.set_lang(language)

    if random:
        random_page = wikipedia.random(pages=1)
        page=random_page
        content=wikipedia.page(page).content
    else:
        search=wikipedia.search(kwargs["keyword"])
        page=search[0]
        content=wikipedia.page(page).content

    return page, content

def translate_text(text: str, target_lang: str, api_key: str=API_KEY)->str:
    """
    The goal of this function is to
    translate a given sentence from 
    one language to the other
    
    Arguments:
        -text: str: The text to be 
        translated
        -source_lang: The language 
        of the sentence to be translated
        -target_lang: str: The language in
        which the sentence should be translated
    Returns:
        -translation: str: The translated_language
    """

    translator = deepl.Translator(API_KEY)

    result = translator.translate_text(text, target_lang=target_lang) 
    translation = result.text

    return translation

def treat_article(article: str)->List[str]:
    """
    The goal of this function is 
    to modify articles one by one
    by removing special characters
    
    Arguments:
        -article: str: The article 
        to be modified
    Returns:
        -modified_article: List[str]:
        The list of modified sentences 
    """
    article=article.split("\n")
    article=[sentence for sentence in article if sentence != '' and "=" not in sentence]
    article=" ".join(article)
    article=article.split(".")
    modified_article=[sentence.strip() for sentence in article]
    modified_article=[re.sub(r"[^a-zA-Z0-9\sàáâãäçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ']", '', sentence) for sentence in modified_article if len(sentence)>35]

    return modified_article

def translate_content(df: pd.DataFrame, output_language: str="EN-GB")->pd.DataFrame:
    df["content_translated"]=df["content"].apply(lambda texte: translate_text(text=texte, target_lang=output_language))
    return df

@functions_framework.http
def wikipedia_etl(request):
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