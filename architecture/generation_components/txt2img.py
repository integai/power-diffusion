import torch
import torch.nn as nn
from samplers import sampler_list as sm
import random
from tqdm import tqdm

class generate_txt2img(nn.Module):
    def __init__(self, model, prompt, guidance_scale, width, height, steps, sampler='dpmsolver++', seed=None, negative_prompt=None):
        super(generate_txt2img, self).__init__()
        self.model = model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            raise ValueError("Model is None. Please provide a valid model.")
        if negative_prompt is None:
            negative_prompt = ""
        image = torch.randn(1, 3, height, width, device=device)
        if seed is None or seed < 10000:
            seed = random.randint(10000, 99999)
        torch.manual_seed(seed)
        sampler_var = sm.sampler_choose(model=model, sampler=sampler)
        
        for _ in tqdm(range(steps), desc="Training Progress"):  # Initialize tqdm
            z = torch.randn_like(image)
            positive_loss = self.model(image, z, prompt, guidance_scale)
            if negative_prompt:
                negative_loss = self.model(image, z, negative_prompt, None, guidance_scale)
                loss = positive_loss - negative_loss
            else:
                loss = positive_loss
            self.model.zero_grad()
            loss.backward()
            image = self._update_image(sampler_var, sampler, image, image.grad)
            image.clamp_(0, 1)
        self.generated_image = image[0].cpu()
    
    def _update_image(self, sampler_var, sampler, image, grad):
        return sampler_var(sampler, image, grad)
