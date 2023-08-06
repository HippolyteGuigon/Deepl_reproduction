import json
from eventregistry import *
from Deepl_reproduction.configs.confs import load_conf, clean_params

main_params = load_conf("configs/main.yml",include=True)
main_params = clean_params(main_params)

EVENTREGISTRY_API_KEY=main_params["EVENTREGISTRY_API_KEY"]

def get_eventregistry_article()->dict:
    """
    The goal of this function is
    to go scrap articles from the
    eventregistry API
    
    Arguments:
        -None
    Returns:
        -content: dict: The dictionnary
        containing the id, title and text 
        of the article 
    """

    content={"uri": [], "title": [], "language":[], "text": []}
    er = EventRegistry(apiKey = EVENTREGISTRY_API_KEY)

    usUri = er.getLocationUri("USA")    # = http://en.wikipedia.org/wiki/United_States

    q = QueryArticlesIter(
        keywords = QueryItems.OR([""]),
        minSentiment = 0.4,
        sourceLocationUri = usUri,
        dataType = ["news", "blog"])
    
    for article in q.execQuery(er, sortBy = "date", maxItems = 500):
        if article["uri"] in content["uri"] or article["lang"] != "en":
            continue 
        content["uri"].append(article["uri"])
        content["title"].append(article["title"])
        content["language"].append(article["lang"])
        content["text"].append(article["body"])

    return content