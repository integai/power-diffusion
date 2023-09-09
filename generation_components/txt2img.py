import torch
from torchvision.transforms import functional as F
from diffusers import DDIMPipeline
from safetensors.torch import load_file
import os
from samplers import sampler_list as sm

def generate_image(model, prompt, guidance_scale, width, height, steps, sampler='dpmsolver++', clip_skip=False, seed=None, negative_prompt=None, output_path="generated_image.png"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    if negative_prompt is None or not negative_prompt:
        negative_prompt = ""

    # Initialize the image with random noise
    image = torch.randn(1, 3, height, width).to(device)

    # Set a random seed for reproducibility if seed is provided (fix seed assignment)
    if seed is not None:
        torch.manual_seed(seed)

    # Create an instance of DPM_Solver
    if sampler == 'dpmsolver++':
        sampler_var = sm.dpm_solver_plus(model=model)
    elif sampler == 'dpmsolver':
        sampler_var = sm.dpm_solver(model=model)

    for step in range(steps):
        # Generate a random vector for diversity
        z = torch.randn_like(image)

        # Calculate the loss with the positive prompt
        positive_loss = model(image, z, prompt, guidance_scale)

        # Calculate the loss with the negative prompt if provided
        if negative_prompt:
            negative_loss = model(image, z, negative_prompt, None, guidance_scale)
            loss = positive_loss - negative_loss
        else:
            loss = positive_loss

        # Update the image using gradient ascent
        image.grad = None
        loss.backward()

        # Use the DPM_Solver to update the image
        image.data = sampler_var(sampler, image, image.grad)

        # Clip the image values to be in the range [0, 1]
        image.data = torch.clamp(image.data, 0, 1)

    # Convert the generated tensor to a PIL image
    generated_image = F.to_pil_image(image[0].cpu())

    # Save the generated image
    generated_image.save(output_path)

    return generated_image