import os
import requests
from tqdm import tqdm 

def download_model_sd(save_path='', model='', model_ext=''):
    ext_list_available = ['.ckpt', '.safetensors']
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    if "." not in model_ext:
        model_ext = "." + model_ext
    if not model_ext in ext_list_available:
        raise ValueError(f"Model extension {model_ext} is not available. Available extensions: ckpt, safetensors.")
    if model == 'sd':
        sd_model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned" + model_ext
        output_ext_sd = model_ext
        save_path = os.path.join(save_path, "sd-v1-5-pruned" + model_ext)
        response = requests.get(sd_model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # Block size for updating the progress bar
        with open(save_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"Downloading SD") as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
        print(f"SD Successfully installed in: {save_path}")
        return save_path
    elif model == 'sdxl':
        sdxl_model_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
        output_ext_sdxl = ".safetensors"
        save_path = os.path.join(save_path, "sd_xl_base_1.0" + output_ext_sdxl)
        # Download the Stable Diffusion XL model with progress bar
        response = requests.get(sdxl_model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # Block size for updating the progress bar
        with open(save_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"Downloading SDXL") as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
        print(f"SDXL Successfully installed in: {save_path}")
        return save_path
    else:
        raise ValueError('Model "'  + model +'" does not exist')