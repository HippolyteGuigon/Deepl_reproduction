import wikipedia 
from typing import List

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