import streamlit as st 

st.title("Deepl Reproduction project")
st.write("Welcome page")

options = ["English", "Russian", "Chinese"]

# Sélectionnez une option à partir de la liste déroulante
selection = st.selectbox("Select the language you want to translate french to :", options)

# Affichez la sélection
st.write("You have chosen to translate from French to :", selection)