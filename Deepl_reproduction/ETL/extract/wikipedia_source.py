import wikipedia 
from typing import List

def get_wikipedia_article(keyword: str , language:str="fr")->List[str]:
    """
    The goal of this function is to 
    retrieve a all wikipedia article 
    given a keyword
    
    Arguemnts:
        -keyword: str: The keyword used
        for searching a wikipedia article
        -language: str: The wikipedia version
        in which the given keyword will have to
        be searched
    Returns:
        -summary: str: The summary of the article
        -article: str: The all article
    """

    wikipedia.set_lang(language)

    search=wikipedia.search(keyword)
    summary=wikipedia.summary(search[0])
    content=wikipedia.page(search[0]).content

    return summary, content