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

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_model_dim: int):
        """
        The goal of this class
        is to generate positional
        embedding so that word order
        in the original sequence is 
        maintained
        
        Arguments:
            -max_seq_len: int: The maximum 
            number of words in each sequence
            that can be put in the embedding matrix
            -embed_model_dim: int: The dimension
            of the embedding generated by the 
            model
        Returns:
            -None
        """

        super(PositionalEmbedding,self).__init__()
        self.embed_dim=embed_model_dim

        pe=torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos,i]=math.sin(pos/(10000**((2*i)/self.embed_dim)))
                pe[pos, i+1]=math.cos(pos/(10000**((2*(i+1))/self.embed_dim)))

        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x:torch.tensor)->torch.tensor:
        """
        The goal of this function
        is, given an input embedded
        vector, to add to it the positional 
        embedding
        
        Arguments:
            -x: torch.tensor: The input vector
        Returns:
            -x: torch.tensor: The same
            tensor once positional encoding was
            added
        """

        x=x*math.sqrt(self.embed_dim)
        seq_len=x.size(1)
        x+=torch.autograd.Variable(self.pe[:,:seq_len],requires_grad=False)
        
        return x
    
class MuliHeadAttention(nn.Module):
    def __init__(self,embed_dim:int=512,n_heads:int=8):
        super(MuliHeadAttention,self).__init__()

        self.embed_dim=embed_dim
        self.n_heads=n_heads 
        self.single_head_dim=int(self.embed_dim/self.n_heads)

        self.query_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.key_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.value_matrix=nn.Linear(self.single_head_dim,self.single_head_dim,bias=False)
        self.out=nn.Linear(self.n_heads*self.single_head_dim,self.embed_dim)