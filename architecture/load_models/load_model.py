import torch
import torch.nn as nn
from safetensors.torch import load_file

def load_model(model_path: str):
    if ".ckpt" in model_path:
        try:
            checkpoint = torch.load(model_path)
            model = nn.Module()
            model.load_state_dict(checkpoint['state_dict'])
            print("Model has been successfully loaded.")
            print(checkpoint['state_dict'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model = model.eval()
            return model
        except FileNotFoundError:
            print(f"Error while loading model: File '{model_path}' not found.")
        except Exception as e:
            print(f"Error while loading model: {str(e)}")
    elif ".safetensors" in model_path:
        try:
            model = load_file(model_path)
            print(model)
            return model
        except FileNotFoundError:
            print(f"Error while loading model: File '{model_path}' not found.")
        except Exception as e:
            print(f"Error while loading model: {str(e)}")
    else:
        raise ValueError('Your model is unsupported.\nUse model with ".ckpt" or ".safetensors" extensions.')

def build_layers(loaded_model, pre_trained_model):
    # Create a new Sequential container
    new_model = nn.Sequential()

    # Add layers from the loaded_model to the new_model
    for name, layer in loaded_model.named_children():
        new_model.add_module(name, layer)

    return new_model