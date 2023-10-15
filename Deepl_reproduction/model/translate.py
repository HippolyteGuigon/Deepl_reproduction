import torch
import os
import glob
import torch.nn.functional as F
import youtokentome
import math
import sys
sys.path.insert(0, "./")

from Deepl_reproduction.model.model import Transformer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BPE Model

def launch_model(language: str="en")->None:
    assert language in ["en", "ja"], "Language must be either en or ja"

    global bpe_model_path, bpe_model, model_path, checkpoint, model

    if language=="en":
        bpe_model_path=glob.glob(os.path.join(os.getcwd(),"Deepl_reproduction/model/english_data/bpe*"))[0]
        if not os.path.exists(
        "Deepl_reproduction/model/english_data/english_transformer_checkpoint.pth.tar"
    ):
            
            model_path = glob.glob("Deepl_reproduction/model/english_data/*.pth.tar")[0]
            checkpoint = torch.load(model_path)
            model = checkpoint["model"].to(device)
        else:
            checkpoint = torch.load(
                "Deepl_reproduction/model/english_data/english_transformer_checkpoint.pth.tar"
            )
            model = checkpoint["model"].to(device)
    elif language=="ja":
        bpe_model_path=glob.glob(os.path.join(os.getcwd(),"Deepl_reproduction/model/japanese_data/bpe*"))[0]
        if not os.path.exists(
        "Deepl_reproduction/model/english_data/japanese_transformer_checkpoint.pth.tar"
    ):
            
            model_path = glob.glob("Deepl_reproduction/model/japanese_data/*.pth.tar")[0]
            checkpoint = torch.load(model_path)
            model = checkpoint["model"].to(device)
        else:
            checkpoint = torch.load(
                "Deepl_reproduction/model/english_data/japanese_transformer_checkpoint.pth.tar"
            )
            model = checkpoint["model"].to(device)
    bpe_model = youtokentome.BPE(model=bpe_model_path)

    # Transformer model

    model.eval()


def translate(source_sequence, language: str="en",beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """

    assert language in ["en", "ja"], "Traduction is only available for english and japanese !"

    launch_model(language=language)
    
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        vocab_size = bpe_model.vocab_size()

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(
                source_sequence,
                output_type=youtokentome.OutputType.ID,
                bos=False,
                eos=False,
            )
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(
                0
            )  # (1, source_sequence_length)
        else:
            encoder_sequences = source_sequence
        encoder_sequences = encoder_sequences.to(device)  # (1, source_sequence_length)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(
            device
        )  # (1)

        # Encode
        encoder_sequences = model.encoder(
            encoder_sequences=encoder_sequences,
            encoder_sequence_lengths=encoder_sequence_lengths,
        )  # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[bpe_model.subword_to_id("<BOS>")]]).to(
            device
        )  # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device)  # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            decoder_sequences = model.decoder(
                decoder_sequences=hypotheses,
                decoder_sequence_lengths=hypotheses_lengths,
                encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                encoder_sequence_lengths=encoder_sequence_lengths.repeat(s),
            )  # (s, step, vocab_size)

            # Scores at this step
            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(
                k, 0, True, True
            )  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat(
                [hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1
            )  # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == bpe_model.subword_to_id(
                "<EOS>"
            )  # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend(
                (top_k_hypotheses_scores[complete] / norm).tolist()
            )

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(
                hypotheses.size(0) * [hypotheses.size(1)]
            ).to(
                device
            )  # (s)

            # Stop if things have been going on for too long
            if step > 100:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        for i, h in enumerate(bpe_model.decode(completed_hypotheses)):
            all_hypotheses.append(
                {"hypothesis": h, "score": completed_hypotheses_scores[i]}
            )

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]
        return best_hypothesis, all_hypotheses


if __name__ == "__main__":
    translation, _ = translate("le chat est en train de dormir sur le canapé")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("le chat est en train de dormir sur le canapé", " ", translation)
    print("\n")
    translation, _ = translate("la voiture est rouge")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("la voiture est rouge", " ", translation)
    print("\n")
    translation, _ = translate("j'habite à paris")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("j'habite à paris", " ", translation)
    print("\n")
    translation, _ = translate("bonjour, je m'appelle hippolyte")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("bonjour, je m'appelle Hippolyte", " ", translation)
    print("\n")
    translation, _ = translate("bonjour, je suis hippolyte")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("bonjour, je suis Hippolyte", " ", translation)
    print("\n")
    translation, _ = translate("je suis né en 1997")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("je suis né en 1997", " ", translation)
    print("\n")
    translation, _ = translate("j'ai deux frères")
    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()
    print("j'ai deux frères", " ", translation)
