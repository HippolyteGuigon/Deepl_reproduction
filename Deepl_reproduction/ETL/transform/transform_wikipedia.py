import re 
import pandas as pd
from typing import List 
from Deepl_reproduction.ETL.transform.traduction import translate_text

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

def translate_content(df: pd.DataFrame, input_language: str="fr", output_language: str="en")->pd.DataFrame:
    df["content_translated"]=df["content"].apply(lambda texte: translate_text(text=texte, source_lang=input_language, target_lang=output_language))
    return df