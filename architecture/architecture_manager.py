from download_components import download_sd_base as sdl
from load_models.load_model import load_model , build_layers
import os
from other_components import model_manager
import time
import torch
from generation_components.txt2img import generate_txt2img

base_models_settings = {
    'save_path': 'models',
    'extensions': 'safetensors',
    'type': 'sdxl', # 2 types: sd/sdxl
    'install': True,
}


generation_settings = {
    'model_path': '',
    'prompt' : '',
    'negative_prompt': '',
    'guidance_scale': 9,
    'width': 512,
    'height': 768,
    'steps': 30,
    'sampler': 'dpmsolver++',
    'seed': None
}

bms = base_models_settings
gs= generation_settings
def pre_gen_process():
    save_path = 'models'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if gs['model_path'] is not None:
        if bms['install'] is True:
            sdl.download_model_sd(save_path=bms['save_path'], model=bms['type'], model_ext=bms['extensions'])

            quit()
def model_pre_process():
    loaded_model = load_model(model_path=gs['model_path'])
    loaded_model = build_layers(loaded_model=loaded_model)
    return loaded_model
pre_gen_process()
model = model_pre_process()

if model is None:
    print('That fucking model does not working properly.')

generate_txt2img(model=load_model(), prompt=gs['prompt'], 
                 negative_prompt=gs['negative_prompt'], 
                 guidance_scale=gs['guidance_scale'],
                  width=gs['width'], height=gs['height'],
                  steps=gs['steps'], sampler=gs['sampler'],
                  seed=gs['seed'])