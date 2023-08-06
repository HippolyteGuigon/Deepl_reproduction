import functions_framework
from flask import Flask, request
from Deepl_reproduction.ETL.wikipedia_cloud_function import wikipedia_etl_pipeline

@functions_framework.http
def launch_wikipedia_pipeline(request)->None:
    """
    The goal of this function 
    is to launch the wikipedia 
    pipeline ETL process cloud 
    function 
    
    Arguments:
        -requests
    Returns:
        -None
    """
    wikipedia_etl_pipeline(request)

    return "Success"