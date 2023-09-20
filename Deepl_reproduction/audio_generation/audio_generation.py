from gtts import gTTS
import os

def read_text(text: str,lang: str='fr')->None:

    tts = gTTS(text=text, lang='fr')

    tts.save("output.mp3")

    os.system("mpg123 output.mp3") 
    os.remove("output.mp3")