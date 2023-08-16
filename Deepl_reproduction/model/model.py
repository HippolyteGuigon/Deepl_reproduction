import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)

import sys 
sys.path.insert(0,"Deepl_reproduction/pipeline")
from Deepl_reproduction.logs.logs import main
from data_loading import load_all_data, load_data_to_front_database, load_data
from sklearn.model_selection import  train_test_split

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        The goal of this class is
        to embed each word entering
        the model
        
        Arguments:
            -vocab_size: int: The size
            of the vocabulary (in the 
            language)
            -embed_dim: int: The dimension
            of the embedding. In which dimension
            are input words embedded
        """

        super(Embedding, self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_dim)

    def foward(self, x: torch.tensor)->torch.tensor:
        """
        The goal of this function is
        to activate the embedding,
        transforming a word input 
        vector into an embedded 
        output vector 
        
        Arguments:
            -x: torch.tensor: The input
            word to be embedded 
        Returns:
            -out: torch.tensor: The output
            embedded vector 
        """

        out=self.embed(x)
        
        return out