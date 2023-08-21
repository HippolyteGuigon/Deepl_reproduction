import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import logging
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
import sys 
import tensorflow_hub as hub
import torch.optim as optim
from transformers import BertTokenizer

warnings.filterwarnings("ignore")

sys.path.insert(0,"Deepl_reproduction/pipeline")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

from Deepl_reproduction.logs.logs import main
from torch.utils.data import Dataset, DataLoader,TensorDataset
from data_loading import load_all_data, load_data_to_front_database, load_data

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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

    def forward(self, x: torch.tensor)->torch.tensor:
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

    def forward(self, key, query, value, mask=None):
        
        batch_size=key.size(0)
        seq_length=key.size(1)

        seq_length_query=query.size(1)

        key=key.view(batch_size,seq_length,self.n_heads,self.single_head_dim)
        query=query.view(batch_size,seq_length,self.n_heads,self.single_head_dim)
        value=value.view(batch_size,seq_length,self.n_heads,self.single_head_dim)

        k=self.key_matrix(key)
        q=self.query_matrix(query)
        v=self.value_matrix(value)

        q=q.transpose(1, 2)
        k=k.transpose(1, 2)
        v=v.transpose(1,2)

        k_adjusted=k.transpose(-1,-2)
        product=torch.matmul(q,k_adjusted)

        if mask is not None:
            product=product.masked_fill(mask==0,float(1e-20))
        
        product=product/math.sqrt(self.single_head_dim)

        scores=F.softmax(product,dim=-1)

        scores=torch.matmul(scores,v)

        concat=scores.transpose(1, 2).contiguous().view(batch_size,seq_length_query, self.single_head_dim*self.n_heads)

        output=self.out(concat)


        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4,n_heads=8):
        super(TransformerBlock,self).__init__()

        self.attention=MuliHeadAttention(embed_dim,n_heads)

        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        self.feed_forward=nn.Sequential(nn.Linear(embed_dim,expansion_factor*embed_dim),
                                        nn.ReLU(),
                                        nn.Linear(expansion_factor*embed_dim,embed_dim)
                                        )
        
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)


    def forward(self,key, query, value):
        attention_out=self.attention(key,query,value)
        attention_residual_out=attention_out+value 
        norm1_out=self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out=self.feed_forward(norm1_out)

        feed_fwd_residual_out=feed_fwd_out+norm1_out
        norm2_out=self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out
    
class TransformerEncoder(nn.Module):

    def __init__(self,seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder,self).__init__()

        self.embedding_layer=Embedding(vocab_size,embed_dim)
        self.positional_encoder=PositionalEmbedding(seq_len,embed_dim)

        self.layers=nn.ModuleList([TransformerBlock(embed_dim,expansion_factor,n_heads) for i in range(num_layers)])

    def forward(self, x):
        embed_out=self.embedding_layer(x)
        out=self.positional_encoder(embed_out)
        
        for layer in self.layers:
            out=layer(out,out,out)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4,n_heads=8):
        super(DecoderBlock,self).__init__()

        self.attention=MuliHeadAttention(embed_dim,n_heads=8)
        self.norm=nn.LayerNorm(embed_dim)
        self.dropout=nn.Dropout(0.2)
        self.transformer_block=TransformerBlock(embed_dim,expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        attention=self.attention(x, x, x, mask=mask)
        value=self.dropout(self.norm(attention+x))

        out=self.transformer_block(key, query, value)


        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder,self).__init__()

        self.word_embedding=nn.Embedding(target_vocab_size,embed_dim)
        self.position_embedding=PositionalEmbedding(seq_len, embed_dim)

        self.layers=nn.ModuleList([DecoderBlock(embed_dim,expansion_factor=4,n_heads=8) for _ in range(num_layers)])

        self.fc_out=nn.Linear(embed_dim,target_vocab_size)
        self.dropout=nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        x=self.word_embedding(x)
        x=self.position_embedding(x)
        x=self.dropout(x)

        for layer in self.layers:
            x=layer(enc_out,x, enc_out, mask)

        out=F.softmax(self.fc_out(x))


        return out 
    

class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,n_heads=8):
        super(Transformer,self).__init__()

        
        self.target_vocab_size=target_vocab_size
        self.encoder=TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        self.decoder=TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,expansion_factor=expansion_factor, n_heads=n_heads)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape

        trg_mask=torch.tril(torch.ones((trg_len,trg_len))).expand(batch_size, 1, trg_len, trg_len)
        return trg_mask
    
    def decode(self, src, trg):
        trg_mask=self.make_trg_mask(trg)
        enc_out=self.encoder(src)
        out_labels=[]
        batch_size, seq_len = src.shape[0], src.shape[1]

        out=trg

        for i in range(seq_len):
            out=self.decoder(out, enc_out, trg_mask)
            out=out[:,-1,:]
            out=out.argmax(-1)
            out_labels.append(out.item())
            out=torch.unsqueeze(out,axis=0)
        return out_labels
    
    def forward(self, src, trg):
        trg_mask=self.make_trg_mask(trg)
        enc_out=self.encoder(src)
        outputs=self.decoder(trg, enc_out, trg_mask)
        return outputs
    
def fit_transformer(model, max_seq_length, batch_size=32, num_epochs=10, learning_rate=1e-3, device='cpu'):
    load_data_to_front_database()
    df_front_database=load_data()
    df_front_database=df_front_database
    src_sentences=df_front_database["french"].tolist()
    trg_sentences=df_front_database["english"].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',hidden_size=16)  # Utilisez le tokenizer BERT
    
    # Tokenize, encode, and pad source sentences
    src_tokens = [tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, pad_to_max_length=True,truncation=True) for text in src_sentences]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long)
    
    # Tokenize, encode, and pad target sentences
    trg_tokens = [tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, pad_to_max_length=True,truncation=True) for text in trg_sentences]
    trg_tensor = torch.tensor(trg_tokens, dtype=torch.long)
    
    # Create DataLoader
    dataset = TensorDataset(src_tensor, trg_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to the specified device
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for index, (src_batch, trg_batch) in enumerate(train_loader):
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src_batch, trg_batch)
            trg_batch = F.one_hot(trg_batch, num_classes=model.target_vocab_size).float()
            loss = criterion(outputs.view(-1, model.target_vocab_size), trg_batch.view(-1, model.target_vocab_size))
            logging.info(f"Loss was computed and is of: {loss:.5f}")
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logging.warning(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    
    logging.warning("Training finished.")

if __name__=="__main__": 
    model = Transformer(embed_dim=16, src_vocab_size=50000, target_vocab_size=50000, seq_length=64, num_layers=3, expansion_factor=2, n_heads=8)
    fit_transformer(model, max_seq_length=32, batch_size=200, num_epochs=10, learning_rate=1e-3, device='cpu')