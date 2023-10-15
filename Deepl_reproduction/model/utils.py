import torch
import os
import wget
import tarfile
import numpy as np
import shutil
import codecs
import glob
import youtokentome
import math
import sys
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0,os.path.join(os.getcwd(),'Deepl_reproduction/configs'))
sys.path.insert(0,os.path.join(os.getcwd(),'Deepl_reproduction/logs'))

from confs import load_conf, clean_params
from logs import main

main()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_params = load_conf("configs/main.yml", include=True)
main_params=clean_params(main_params)

train_size=main_params["train_size"]

def download_data(data_folder):
    """
    Downloads the training, validation, and test files for WMT '14 en-de translation task.

    Training: Europarl v7, Common Crawl, News Commentary v9
    Validation: newstest2013
    Testing: newstest2014

    The homepage for the WMT '14 translation task, https://www.statmt.org/wmt14/translation-task.html, contains links to
    the datasets.

    :param data_folder: the folder where the files will be downloaded

    """
    train_urls = ["http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                  "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                  "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"]

    print("\n\nThis may take a while.")

    # Create a folder to store downloaded TAR files
    if not os.path.isdir(os.path.join(data_folder, "tar files")):
        os.mkdir(os.path.join(data_folder, "tar files"))
    # Create a fresh folder to extract downloaded TAR files; previous extractions deleted to prevent tarfile module errors
    if os.path.isdir(os.path.join(data_folder, "extracted files")):
        shutil.rmtree(os.path.join(data_folder, "extracted files"))
        os.mkdir(os.path.join(data_folder, "extracted files"))

    # Download and extract training data
    for url in train_urls:
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar files", filename)):
            print("\nDownloading %s..." % filename)
            wget.download(url, os.path.join(data_folder, "tar files", filename))
        print("\nExtracting %s..." % filename)
        tar = tarfile.open(os.path.join(data_folder, "tar files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted files"), members=members)

    # Download validation and testing data using sacreBLEU since we will be using this library to calculate BLEU scores
    print("\n")
    os.system("sacrebleu -t wmt13 -l en-de --echo src > '" + os.path.join(data_folder, "val.en") + "'")
    os.system("sacrebleu -t wmt13 -l en-de --echo ref > '" + os.path.join(data_folder, "val.de") + "'")
    print("\n")
    os.system("sacrebleu -t wmt14/full -l en-de --echo src > '" + os.path.join(data_folder, "test.en") + "'")
    os.system("sacrebleu -t wmt14/full -l en-de --echo ref > '" + os.path.join(data_folder, "test.de") + "'")

    # Move files if they were extracted into a subdirectory
    for dir in [d for d in os.listdir(os.path.join(data_folder, "extracted files")) if
                os.path.isdir(os.path.join(data_folder, "extracted files", d))]:
        for f in os.listdir(os.path.join(data_folder, "extracted files", dir)):
            shutil.move(os.path.join(data_folder, "extracted files", dir, f),
                        os.path.join(data_folder, "extracted files"))
        os.rmdir(os.path.join(data_folder, "extracted files", dir))


def prepare_data(data_folder=os.getcwd(), euro_parl=True, common_crawl=True, news_commentary=True, min_length=3, max_length=100,
                 max_length_ratio=1.5, retain_case=True):
    """
    Filters and prepares the training data, trains a Byte-Pair Encoding (BPE) model.

    :param data_folder: the folder where the files were downloaded
    :param euro_parl: include the Europarl v7 dataset in the training data?
    :param common_crawl: include the Common Crawl dataset in the training data?
    :param news_commentary: include theNews Commentary v9 dataset in the training data?
    :param min_length: exclude sequence pairs where one or both are shorter than this minimum BPE length
    :param max_length: exclude sequence pairs where one or both are longer than this maximum BPE length
    :param max_length_ratio: exclude sequence pairs where one is much longer than the other
    :param retain_case: retain case?
    """

    src_language="french"
    target_language="english"
    src_short="fr"
    trg_short="en"
    # Read raw files and combine
    src = list()
    trg = list()
    files = list()
    
    full_data=pd.read_csv(data_folder)
    X_train=full_data.loc[:np.floor(full_data.shape[0]*train_size),:]
    train_shape=X_train.shape[0]
    X_test=full_data.loc[train_shape:train_shape+(full_data.shape[0]-train_shape)//2:,:]   
    X_val=full_data.loc[train_shape+(full_data.shape[0]-train_shape)//2:,:]   
    
    os.chdir("Deepl_reproduction/model")
    data_folder=os.getcwd()
    src_sentences=X_train[src_language].tolist()
    trg_sentences=X_train[target_language].tolist()
    src_sentences=[str(sentence).lower() for sentence in src_sentences]
    trg_sentences=[str(sentence).lower() for sentence in trg_sentences]

    assert len(src_sentences)==len(trg_sentences), "The two sentence sets do not have the same size"
    src.extend(src_sentences)
    trg.extend(trg_sentences)
    assert len(src_sentences)==len(trg_sentences), "The two sentence sets do not have the same size"

    src_test=X_test[src_language].tolist()
    trg_test=X_test[target_language].tolist()
    src_test=[str(sentence).lower() for sentence in src_test]
    trg_test=[str(sentence).lower() for sentence in trg_test]

    src_val=X_val[src_language].tolist()
    trg_val=X_val[target_language].tolist()
    src_val=[str(sentence).lower() for sentence in src_val]
    trg_val=[str(sentence).lower() for sentence in trg_val]

    with open(os.path.join(data_folder, "test."+src_short), "w", encoding="utf-8") as f:
        f.write("\n".join(src_test))
    with open(os.path.join(data_folder, "test."+trg_short), "w", encoding="utf-8") as f:
        f.write("\n".join(trg_test))
    with open(os.path.join(data_folder, "val."+src_short), "w", encoding="utf-8") as f:
        f.write("\n".join(src_val))
    with open(os.path.join(data_folder, "val."+trg_short), "w", encoding="utf-8") as f:
        f.write("\n".join(trg_val))

    print("\nCombining...")
    # Write to file so stuff can be freed from memory
    print("\nWriting to single files...")
    with open(os.path.join(data_folder, "train."+trg_short), "w", encoding="utf-8") as f:
        f.write("\n".join(trg))
    with open(os.path.join(data_folder, "train."+src_short), "w", encoding="utf-8") as f:
        f.write("\n".join(src))
    with open(os.path.join(data_folder, "train."+src_short+trg_short), "w", encoding="utf-8") as f:
        f.write("\n".join(src + trg))
    del src, trg  # free some RAM
    
    # Perform BPE
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "train."+src_short+trg_short), vocab_size=37000,
                           model=os.path.join(data_folder, "bpe.model"))

    # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

    # Re-read English, French
    print("\nRe-reading single files...")
    with open(os.path.join(data_folder, "train."+trg_short), "r", encoding="utf-8") as f:
        trg = f.read().split("\n")
    with open(os.path.join(data_folder, "train."+src_short), "r", encoding="utf-8") as f:
        src = f.read().split("\n")

    # Filter
    print("\nFiltering...")
    pairs = list()
    for en, de in tqdm(zip(trg, src), total=len(trg)):
        en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
        fr_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
        len_en_tok = len(en_tok)
        len_fr_tok = len(fr_tok)
        if min_length < len_en_tok < max_length and \
                min_length < len_fr_tok < max_length and \
                1. / max_length_ratio <= len_fr_tok / len_en_tok <= max_length_ratio:
            pairs.append((en, de))
        else:
            continue
    print("\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits." % (100. * (
            len(trg) - len(pairs)) / len(trg)))

    # Rewrite files
    trg, src = zip(*pairs)
    print("\nRe-writing filtered sentences to single files...")
    os.remove(os.path.join(data_folder, "train."+trg_short))
    os.remove(os.path.join(data_folder, "train."+src_short))
    os.remove(os.path.join(data_folder, "train."+src_short+trg_short))

    assert len(trg)==len(src)
    print(len(src),len(trg))
    with codecs.open(os.path.join(data_folder, "train."+trg_short), "w", encoding="utf-8") as f:
        f.write("\n".join(trg))

    with codecs.open(os.path.join(data_folder, "train."+src_short), "w", encoding="utf-8") as f:
        f.write("\n".join(src))

    del src, trg, bpe_model, pairs

    print("\n...DONE!\n")


def get_positional_encoding(d_model, max_length=100):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

    return lr


def save_checkpoint(epoch, model, optimizer, prefix='', language="en"):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    assert language in ["en", "ja"], "Language must be either en or ja"

    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    
    if language=="en":
        prefix="english_"
        model_path="Deepl_reproduction/model/english_data"
    elif language=="ja":
        prefix="japanese_"
        model_path="Deepl_reproduction/model/japanese_data"
    filename = prefix + 'transformer_checkpoint.pth.tar'
    logging.info(f"Saving {filename} in {model_path}")
    torch.save(state, os.path.join(model_path,filename))


def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
