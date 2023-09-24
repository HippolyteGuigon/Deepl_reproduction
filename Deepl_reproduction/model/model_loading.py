import torch 
import os
import logging
from google.cloud import storage
from Deepl_reproduction.configs.confs import load_conf, clean_params
from Deepl_reproduction.logs.logs import main

main()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

main_params=load_conf("configs/main.yml",include=True)
main_params=clean_params(main_params)
local_model_path=main_params['checkpoint']

client = storage.Client.from_service_account_json('deepl_api_key.json', project='deepl-reprodution')

def load_model(language: str="english", load_gcp: bool=True, load_best=True, **kwargs)->torch:
    """
    The goal of this function is to load
    the traduction model either from google
    cloud storage or from local file
    
    Arguments:
        -load_gcp: bool: Either the model
        should be loaded from google cloud
        storage
        -load_best: bool: If the model is
        loaded from google cloud storage, 
        should the best model be loaded
    Returns:
        -model: torch: The final model 
        loaded
    """

    save_path='Deepl_reproduction/model'
    
    assert language in ["english", "japanese"], "The model loaded should be either french or japanese !"

    if load_gcp:
        if language=="english":
            bucket = client.get_bucket('english_deepl_bucket')
            if not os.path.exists(os.path.join(save_path,"bpe_english.model")):
                blob = bucket.blob("bpe_english.model")
                blob.download_to_filename(os.path.join(save_path,"bpe_english.model"))

        elif language=="japanese":
            bucket = client.get_bucket('japanese_deepl_bucket')
            if not os.path.exists(os.path.join(save_path,"bpe_japanese.model")):
                blob = bucket.blob("bpe_japanese.model")
                blob.download_to_filename(os.path.join(save_path,"bpe_japanese.model"))

        if load_best:
            blobs = bucket.list_blobs()
            file_names = [blob.name for blob in blobs if "bpe"  not in blob.name]
            loss_gcp=[name.split("model_loss_")[-1].replace('.pth.tar','') for name in file_names]
            loss_gcp=[float(loss.replace("_",".")) for loss in loss_gcp]
            min_loss_gcp=min(loss_gcp)
            best_model_name=[name for name in file_names if str(min_loss_gcp).replace(".", "_") in name][0]
            final_save_path=os.path.join(os.getcwd(),save_path,best_model_name)
            logging.info(f"Downloading {best_model_name} model...")
            blob = bucket.blob(best_model_name)
            blob.download_to_filename(final_save_path)

            
            logging.info(f"Model {best_model_name} successfully downloaded under the path {final_save_path}")
            model=torch.load(final_save_path)
            return model
        
        else:
            if "gcp_model_name" not in kwargs.keys():
                raise ValueError("If you want to load another model than the best, \
                                please enter the name of the model you want under the\
                                'gcp_model_name' argument")
            final_save_path=os.path.join(os.getcwd(),save_path,kwargs['gcp_model_name'])
            logging.info(f"Downloading {kwargs['gcp_model_name']} model...")
            blob = bucket.blob(kwargs['gcp_model_name'])
            blob.download_to_filename(final_save_path)
            logging.info(f"Model {kwargs['gcp_model_name']} successfully downloaded under the path {final_save_path}")
            model=torch.load(final_save_path)
            return model

    else:
        model=torch.load(local_model_path)
        return model