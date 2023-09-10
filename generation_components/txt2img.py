import torch
from torchvision.transforms import functional as F
from samplers import sampler_list as sm

def generate_image(model, prompt, guidance_scale, width, height, steps, sampler='dpmsolver++', clip_skip=False, seed=None, negative_prompt=None, output_path="generated_image.png"):
    if output_path is None:
        output_path = "1.png"
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    if negative_prompt is None or not negative_prompt:
        negative_prompt = ""
    
    image = torch.randn(1, 3, height, width, device=device)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    if sampler == 'dpmsolver++':
        sampler_var = sm.dpm_solver_plus(model=model)
    elif sampler == 'dpmsolver':
        sampler_var = sm.dpm_solver(model=model)
    
    with torch.no_grad():
        for _ in range(steps):
            z = torch.randn_like(image)
            positive_loss = model(image, z, prompt, guidance_scale)
            
            if negative_prompt:
                negative_loss = model(image, z, negative_prompt, None, guidance_scale)
                loss = positive_loss - negative_loss
            else:
                loss = positive_loss
            
            model.zero_grad()
            loss.backward()
            image.data = sampler_var(sampler, image, image.grad)
            image.data.clamp_(0, 1)
    
    generated_image = F.to_pil_image(image[0].cpu())
    generated_image.save(output_path)
    
    return generated_image