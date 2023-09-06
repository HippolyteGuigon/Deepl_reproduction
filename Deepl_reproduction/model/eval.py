import torch
import sacrebleu
from translate import translate
from tqdm import tqdm
from dataloader import SequenceLoader
import youtokentome
from nltk.translate.bleu_score import sentence_bleu 
import codecs
import os

# Use sacreBLEU in Python or in the command-line?
# Using in Python will use the test data downloaded in prepare_data.py
# Using in the command-line will use test data automatically downloaded by sacreBLEU...
# ...and will print a standard signature which represents the exact BLEU method used! (Important for others to be able to reproduce or compare!)
sacrebleu_in_python = True

# Make sure the right model checkpoint is selected in translate.py

# Data loader
test_loader = SequenceLoader(data_folder=os.path.join(os.getcwd(),"Deepl_reproduction/model"),
                             source_suffix="fr",
                             target_suffix="en",
                             split="test",
                             tokens_in_batch=None)
test_loader.create_batches()

model=torch.load("Deepl_reproduction/model/steplast_transformer_checkpoint.pth.tar")

hypothesis = ["Hello, my name is Hippolyte."]

# Exemple de références humaines (plusieurs références sont possibles)
references = [["My name is Hippolyte", "My Hippolyte"]]

# Calcul du score BLEU
bleu_score = sacrebleu.corpus_bleu(hypothesis, references)

# Affichage du score BLEU
print(f"BLEU Score: {bleu_score.score}")

# Evaluate
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
        except RuntimeError:
            continue
    if sacrebleu_in_python:
        print("\n13a tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references]))
        print("\n13a tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
        print("\nInternational tokenization, cased:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
        print("\nInternational tokenization, caseless:\n")
        print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
        print("\n")
    else:
        with codecs.open("translated_test.de", "w", encoding="utf-8") as f:
            f.write("\n".join(hypotheses))
        print("\n13a tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de")
        print("\n13a tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -lc")
        print("\nInternational tokenization, cased:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl")
        print("\nInternational tokenization, caseless:\n")
        os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
        print("\n")
    print(
        "The first value (13a tokenization, cased) is how the BLEU score is officially calculated by WMT (mteval-v13a.pl). \nThis is probably not how it is calculated in the 'Attention Is All You Need' paper, however.\nSee https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191 for more details.\n")
