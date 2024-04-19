import os
from transformers import AutoModel
from dotenv import load_dotenv 

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')

model_storage_folder = 'downloaded_models'

def download_model(model_name, model_storage_folder):
    model_path = os.path.join(model_storage_folder, model_name.replace('/', '_'))
    os.makedirs(model_path, exist_ok=True)

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(model_path)
    
    print(f"Model downloaded and saved in {model_path}")
    return model

model_name = 'meta-llama/Llama-2-7b-chat-hf'
downloaded_model = download_model(model_name, model_storage_folder)
