import mlflow
import mlflow.pytorch
import os
import subprocess
from train import *

mlflow.start_run()

def launch_japanese_training(language:str="ja")->None:
    """
    The goal of this function is to wrap the training
    pipeline into a single function 
    
    Arguments:
        -language: str: The language pipeline to be 
        launched
    Rerturns:
        -None
    """
    subprocess.run(["python3", "Deepl_reproduction/model/train.py", "--language", language])


launch_japanese_training()

mlflow.end_run()