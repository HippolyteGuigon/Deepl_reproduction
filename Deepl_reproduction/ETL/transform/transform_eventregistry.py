import re 
from typing import Dict

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