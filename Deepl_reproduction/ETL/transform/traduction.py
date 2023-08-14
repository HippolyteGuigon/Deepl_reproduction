import requests
import deepl
import os
from google.cloud import translate_v2 as translate
from Deepl_reproduction.configs.confs import load_conf, clean_params

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "deepl_api_key.json"
API_KEY=main_params["API_KEY"]

def translate_text(text: str, target_lang: str, api_key: str=API_KEY)->str:
    """
    The goal of this function is to
    translate a given sentence from 
    one language to the other
    
    Arguments:
        -text: str: The text to be 
        translated
        -source_lang: The language 
        of the sentence to be translated
        -target_lang: str: The language in
        which the sentence should be translated
    Returns:
        -translation: str: The translated_language
    """
    client = translate.Client()
    
    try:
        translator = deepl.Translator(API_KEY)
        result = translator.translate_text(text, target_lang=target_lang) 
        translation = result.text
    except Exception as e:
        if "quota" in str(e).lower():
            translated_text = client.translate(text, source_language='fr', target_language=target_lang)
            translation=translated_text["translatedText"]

    return translation