import re 
from typing import List 

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