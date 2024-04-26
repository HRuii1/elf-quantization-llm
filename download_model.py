import os
import torch
from transformers import AutoModel
from dotenv import load_dotenv 
from google.cloud import storage

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


def download_model_to_gcs(model_name, bucket_name):
    # Initialize GCS client and get bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    model_storage_folder = 'downloaded_models'
    safe_model_name = model_name.replace('/', '_')
    model_path = os.path.join(model_storage_folder, safe_model_name)
    os.makedirs(model_path, exist_ok=True)

    local_model_bin_path = os.path.join(model_path, "pytorch_model.bin")
    local_config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(local_model_bin_path) or not os.path.exists(local_config_path):
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(model_path)
        print(f"Model and configuration saved locally at {model_path}")

    blob_bin = bucket.blob(f"{safe_model_name}/pytorch_model.bin")
    blob_bin.upload_from_filename(local_model_bin_path)
    print(f"Model weights uploaded to GCS at {blob_bin.public_url}")

    blob_config = bucket.blob(f"{safe_model_name}/config.json")
    blob_config.upload_from_filename(local_config_path)
    print(f"Model config uploaded to GCS at {blob_config.public_url}")

    os.remove(local_model_bin_path)
    os.remove(local_config_path)
    os.rmdir(model_path)

    return model




from transformers import AutoConfig
import os
from google.cloud import storage

def download_config_to_gcs(model_name, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Local directory structure for model
    model_storage_folder = 'downloaded_models'
    safe_model_name = model_name.replace('/', '_')
    model_path = os.path.join(model_storage_folder, safe_model_name)
    os.makedirs(model_path, exist_ok=True)  # Ensure directory exists

    # Local path for the config.json
    local_config_path = os.path.join(model_path, "config.json")

    # Download and save config locally
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(model_path)
    print(f"Configuration saved locally at {local_config_path}")

    # Upload config to GCS
    blob_config = bucket.blob(f"{safe_model_name}/config.json")
    blob_config.upload_from_filename(local_config_path)
    print(f"Configuration file uploaded to GCS at {blob_config.public_url}")

    # Optional: Cleanup local config file after upload
    os.remove(local_config_path)

model_name = 'facebook/opt-125m'  # Example model, replace with your model name
download_config_to_gcs(model_name, gcs_bucket_name)


# model_name = 'openai-community/gpt2'
# downloaded_model = download_model_to_gcs(model_name, gcs_bucket_name)

#downloaded_model = download_model(model_name, model_storage_folder)
