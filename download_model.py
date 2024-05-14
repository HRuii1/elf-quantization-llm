import os
import torch
from transformers import AutoModel
from dotenv import load_dotenv 
from google.cloud import storage
from transformers import AutoConfig

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')
gcs_bucket_name = os.environ.get('GCS_BUCKET_NAME') 

model_storage_folder = 'downloaded_models'

def download_model(model_name, model_storage_folder):
    model_path = os.path.join(model_storage_folder, model_name.replace('/', '_'))
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, "pytorch_model.bin")

    model = AutoModel.from_pretrained(model_name)
    torch.save(model.state_dict(), model_path)
    
    print(f"Model downloaded and saved in {model_path}")
    return model


def download_config_to_gcs(model_name, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Local directory structure for model
    model_storage_folder = 'downloaded_models'
    safe_model_name = model_name.replace('/', '_')
    model_path = os.path.join(model_storage_folder, safe_model_name)
    os.makedirs(model_path, exist_ok=True)

    local_config_path = os.path.join(model_path, "config.json")

    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(model_path)
    print(f"Configuration saved locally at {local_config_path}")

    blob_config = bucket.blob(f"{safe_model_name}/config.json")
    blob_config.upload_from_filename(local_config_path)
    print(f"Configuration file uploaded to GCS at {blob_config.public_url}")

    # Cleanup local config file after upload
    os.remove(local_config_path)

    
def download_model_to_gcs(model_name, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    model_storage_folder = 'downloaded_models'
    model_path = os.path.join(model_storage_folder, model_name.replace('/', '_'))
    os.makedirs(model_path, exist_ok=True)
    local_model_path = os.path.join(model_path, "pytorch_model.bin")

    model = AutoModel.from_pretrained(model_name)
    
    # Save model to local path 
    torch.save(model.state_dict(), local_model_path)
    print(f"Model state dict saved locally at {local_model_path}")
    # Upload to gcs
    blob = bucket.blob(f"{model_name.replace('/', '_')}/pytorch_model.bin")
    blob.upload_from_filename(local_model_path)
    print(f"Model uploaded to GCS at {blob.public_url}")
    # Delete local path
    os.remove(local_model_path)
    os.rmdir(model_path)

    return model

model_name = 'meta-llama/Llama-2-7b-chat-hf'
download_config_to_gcs(model_name, gcs_bucket_name)
downloaded_model = download_model_to_gcs(model_name, gcs_bucket_name)


#downloaded_model = download_model(model_name, model_storage_folder)
