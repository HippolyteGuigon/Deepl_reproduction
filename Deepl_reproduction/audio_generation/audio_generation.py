from gtts import gTTS
from pydub import AudioSegment
import os

def read_text(text: str,lang: str='fr')->None:

    assert lang in ['fr', 'ja', 'en'], "Only French, English and Japanese are taken into account"

    tts = gTTS(text=text, lang=lang)

    tts.save("output.mp3")

    os.system("mpg123 output.mp3") 

def audio_save(text: str,lang: str='fr')->None:
    assert lang in ['fr', 'ja', 'en'], "Only French, English and Japanese are taken into account"

    tts = gTTS(text=text, lang=lang)

    tts.save("output.mp3")

    audio_mp3 = AudioSegment.from_mp3("output.mp3")

    # Convertir le fichier MP3 en WAV
    audio_mp3.export('output.wav', format="wav")
    os.remove("output.mp3")