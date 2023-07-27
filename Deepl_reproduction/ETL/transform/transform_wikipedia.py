import re 
from typing import List 

def treat_article(article: str)->List[str]:
    article=article.split("\n")
    article=[sentence for sentence in article if sentence != '' and "=" not in sentence]
    article=" ".join(article)
    article=article.split(".")
    article=[sentence.strip() for sentence in article]

    return article