import re 
import pandas as pd
from typing import Dict
from Deepl_reproduction.ETL.transform.traduction import translate_text

def text_cleaning(dico_article: Dict)->Dict:
    """
    The goal of this function is to
    clean all titles and text that have
    just been scrapped
    
    Arguments:
        -dico_article: Dict: The textual
        data that have just been scrapped
    Returns:
        -cleaned_dico_article: The dictionnary
        after having been cleaned
    """

    dico_article["title"]=[re.sub(r"\(.*?\)", "", text) for text in dico_article["title"]]
    dico_article["text"]=[re.sub(r"\(.*?\)", "", text) for text in dico_article["text"]]

    dico_article["title"]=[re.sub(r"[^a-zA-Z0-9\sàáâãäçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ']", '', sentence) for sentence in dico_article["title"]]
    dico_article["text"]=[re.sub(r"[^a-zA-Z0-9\sàáâãäçèéêëìíîïñòóôõöùúûüýÿÀÁÂÃÄÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝ']", '', sentence) for sentence in dico_article["text"]]

    return dico_article

def translate_content(df: pd.DataFrame, output_language: str="FR")->pd.DataFrame:
    """
    The goal of this function is to
    translate the content of the processed
    DataFrame in another language
    
    Arguments:
        -df: pd.DataFrame: The DataFrame
        cointaining the sentences to be
        translated
        -output_language: str: The language
        in which the sentences composing the
        dataframe should be translated in
    Returns:
        -processed_df: The DataFrame with 
        appropriate traductions
    """

    df["title_processed"]=df["title"].apply(lambda title: translate_text(text=title, target_lang=output_language))
    df["text"]=df["text"].apply(lambda texte: texte.split("\n"))
    df["text"]=df["text"].apply(lambda liste: [sentence for sentence in liste if sentence.strip()!=""])
    df=df.explode("text")
    df["text_processed"]=df["text"].apply(lambda texte: translate_text(text=texte, target_lang=output_language))
    
    df["uri"]=df["uri"].astype(int)
    df["title"]=df["title"].astype(str)
    df["title_processed"]=df["title_processed"].astype(str)
    df["text"]=df["text"].astype(str)
    df["text_processed"]=df["text_processed"].astype(str)
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    
    return df