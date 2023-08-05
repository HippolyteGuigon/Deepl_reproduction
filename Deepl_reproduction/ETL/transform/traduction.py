import requests
import deepl
from Deepl_reproduction.configs.confs import load_conf, clean_params

main_params = load_conf("configs/main.yml", include=True)
main_params = clean_params(main_params)

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

    translator = deepl.Translator(API_KEY)

    result = translator.translate_text(text, target_lang=target_lang) 
    translation = result.text

    return translation