from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import sys 
sys.path.insert(0,"Deepl_reproduction/pipeline")
from Deepl_reproduction.logs.logs import main
from data_loading import load_all_data, load_data_to_front_database, load_data
from sklearn.model_selection import  train_test_split

class Translation_Model:
    """
    The goal of this class 
    is to create the transformer
    model that will make the translations
    from one language to another 
    """
    def __init__(self):
        pass 

    def get_data(self)->None:
        load_data_to_front_database()
        self.data=load_data()