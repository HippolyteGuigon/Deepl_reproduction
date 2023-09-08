import streamlit as st
import glob
from Deepl_reproduction.model.model_loading import load_model

st.title("Deepl Reproduction project")
st.write("Welcome page")

options = [" ", "English", "Russian", "Chinese"]

# Sélectionnez une option à partir de la liste déroulante
selection = st.selectbox(
    "Select the language you want to translate french to :", options
)

# Affichez la sélection
if selection=="English":
    st.write("You have chosen to translate from French to :", selection)
    if len(glob.glob('Deepl_reproduction/model/*.pth.tar'))==0:
        st.write("Loading model...")
        load_model()
    else:
        pass
elif selection=="Russian":
    st.write("The traduction from french to Russian is not available yet")
elif selection=="Chinese":
    st.write("The traduction from french to Chinese is not available yet")

from Deepl_reproduction.model.translate import translate

texte_a_traduire = st.text_input("Please enter here the french sentence you wish to translate :", "")

if st.button("Translate"):
    # Ajoutez ici le code pour effectuer la traduction
    # Par exemple, vous pouvez utiliser une bibliothèque de traduction comme translate-python

    # Exemple de code de traduction fictif :
    texte_traduit = translate(texte_a_traduire)

    # Affichez le texte traduit
    st.write(texte_a_traduire,"      translated to:    ", texte_traduit)