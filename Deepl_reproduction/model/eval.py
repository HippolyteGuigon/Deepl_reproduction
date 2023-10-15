import torch
import sacrebleu
import logging
import codecs
import os
import youtokentome
import sys
import numpy as np
from tqdm import tqdm
from translate import launch_model, translate
from dataloader import SequenceLoader
from nltk.translate.bleu_score import sentence_bleu

# Use sacreBLEU in Python or in the command-line?
# Using in Python will use the test data downloaded in prepare_data.py
# Using in the command-line will use test data automatically downloaded by sacreBLEU...
# ...and will print a standard signature which represents the exact BLEU method used! (Important for others to be able to reproduce or compare!)

def evaluation(language: str="en")->float:
    sacrebleu_in_python = True
    launch_model(language=language)
    assert language in ["en", "ja"], "Language must be either en or ja"
    if language=="en":
        data_folder="Deepl_reproduction/model/english_data"
        with open(os.path.join(data_folder,"test.fr"), 'r', encoding="utf-8") as f:
            french=f.read().split("\n")
        with open(os.path.join(data_folder,"test.en"), 'r', encoding="utf-8") as f:
            english=f.read().split("\n")
        assert len(french)==len(english), "French and English test size must have the same length"
        random_indexes=np.unique(np.random.randint(0,len(french),size=100))
        french, english = [french[i] for i in random_indexes], [english[i] for i in random_indexes]
        with open(os.path.join(data_folder,"personnal_test.fr"),"w",encoding="utf-8") as f:
            f.write("\n".join(french))
        with open(os.path.join(data_folder,"personnal_test.en"),"w",encoding="utf-8") as f:
            f.write("\n".join(english))
        
        test_loader = SequenceLoader(data_folder=os.path.join(os.getcwd(),data_folder),
                                    source_suffix="fr",
                                    target_suffix="en",
                                    split="personnal_test",
                                    tokens_in_batch=None)
        test_loader.create_batches()

        with torch.no_grad():
            hypotheses = list()
            references = list()
            for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
                    tqdm(test_loader, total=test_loader.n_batches)):
                try:
                    hypotheses.append(translate(source_sequence=source_sequence,
                                            beam_size=4,
                                            length_norm_coefficient=0.6)[0])
                    references.extend(test_loader.bpe_model.decode(target_sequence.tolist()))

                    hypotheses=[sentence.replace("<BOS>","").replace("<EOS>","").strip().lower() for sentence in hypotheses]
                    references=[sentence.replace("<BOS>","").replace("<EOS>","").strip().lower() for sentence in references]
                except RuntimeError:
                    continue
            
            bleu_score=sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True).score
            bleu_score/=100

            return bleu_score

    elif language=="ja":
        data_folder="Deepl_reproduction/model/japanese_data"
        with open(os.path.join(data_folder,"test.fr"), 'r', encoding="utf-8") as f:
            french=f.read().split("\n")
        with open(os.path.join(data_folder,"test.ja"), 'r', encoding="utf-8") as f:
            japanese=f.read().split("\n")
        assert len(french)==len(japanese), "French and Japanese test size must have the same length"
        random_indexes=np.unique(np.random.randint(0,len(japanese),size=100))
        french, japanese = [french[i] for i in random_indexes], [japanese[i] for i in random_indexes]
        with open(os.path.join(data_folder,"personnal_test.fr"),"w",encoding="utf-8") as f:
            f.write("\n".join(french))
        with open(os.path.join(data_folder,"personnal_test.ja"),"w",encoding="utf-8") as f:
            f.write("\n".join(japanese))
        test_loader = SequenceLoader(data_folder=os.path.join(os.getcwd(),data_folder),
                                    source_suffix="fr",
                                    target_suffix="ja",
                                    split="personnal_test",
                                    tokens_in_batch=None)
        test_loader.create_batches()

        with torch.no_grad():
            hypotheses = list()
            references = list()
            for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
                    tqdm(test_loader, total=test_loader.n_batches)):
                try:
                    hypotheses.append(translate(source_sequence=source_sequence, language=language,
                                            beam_size=4,
                                            length_norm_coefficient=0.6)[0])
                    references.extend(test_loader.bpe_model.decode(target_sequence.tolist()))

                    hypotheses=[sentence.replace("<BOS>","").replace("<EOS>","").strip().lower() for sentence in hypotheses]
                    references=[sentence.replace("<BOS>","").replace("<EOS>","").strip().lower() for sentence in references]
                except RuntimeError:
                    continue

            bleu_score=sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True).score
            bleu_score/=100
            
            return bleu_score