import streamlit as st
import sys
import glob
import os
from Deepl_reproduction.model.model_loading import load_model
from Deepl_reproduction.audio_generation.audio_generation import read_text, audio_save

sys.path.insert(0, "./Deepl_reproduction")
st.title("Deepl Reproduction project")
st.write("Welcome page")

options = [" ", "English", "Russian", "Japanese"]

# Sélectionnez une option à partir de la liste déroulante
selection = st.selectbox(
    "Select the language you want to translate french to :", options
)

# Affichez la sélection
if selection == "English":
    st.write("You have chosen to translate from French to :", selection)
    if len(glob.glob("Deepl_reproduction/model/*.pth.tar")) == 0:
        st.write("Loading model...")
        load_model(language="english")
    else:
        pass

elif selection == "Japanese":
    if len(glob.glob("Deepl_reproduction/model/*.pth.tar")) == 0:
        st.write("Loading model...")
        load_model(language="japanese")
    else:
        pass

elif selection == "Russian":
    st.write("The traduction from french to russian is not available yet")


texte_a_traduire = st.text_input(
    "Please enter here the french sentence you wish to translate :", ""
)

if st.button("Translate"):
    # Ajoutez ici le code pour effectuer la traduction
    # Par exemple, vous pouvez utiliser une bibliothèque de traduction comme translate-python

    # Exemple de code de traduction fictif :
    from Deepl_reproduction.model.translate import translate

    if selection=="English":
        translation, _ = translate(texte_a_traduire,language="en")
    elif selection=="Japanese":
        translation, _ = translate(texte_a_traduire,language="ja")

    translation = translation.replace("<BOS>", "").replace("<EOS>", "").strip()
    translation = translation.capitalize()

    st.session_state.translation = translation

    # Affichez le texte traduit
    st.write(texte_a_traduire)
    st.write("has been translated to: ")
    st.write(translation)

if st.button("Read translation audio"):

    if os.path.exists("output.wav"):
        os.remove("output.wav")

    translation=st.session_state.translation
    if selection=="English":
        audio_save(translation, lang='en')
    elif selection=="Japanese":
        audio_save(translation, lang='ja')

    audio_wav = open('output.wav', "rb").read()
    st.audio(audio_wav, format="audio/wav")
