
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from timm.utils import ModelEmaV3 
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm
from unet import Unet
from vae import VAE
from scheduler import DDPM_Scheduler


def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(15,2))
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.squeeze(0)), 
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t.clamp(0, 1) * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    ])

    for i, ax in enumerate(axes.flat):
        img = reverse_transforms(images[i])
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def inference(checkpoint_path: str=None,
              num_time_steps: int=1000,
              ema_decay: float=0.999, ):
    checkpoint = torch.load(checkpoint_path)
    model = Unet().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0,15,50,100,200,300,400,550,700,999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(1, 3, 64, 64)
        for t in reversed(range(1, num_time_steps)):    
            t = [t]
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            z = (1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(), torch.tensor(t).cuda()).cpu())
            if t[0] in times:
                images.append(z)
            e = torch.randn(1, 3, 64, 64)
            z = z + (e*torch.sqrt(scheduler.beta[t]))

        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),torch.tensor([0]).cuda()).cpu())
        images.append(x)
        display_reverse(images)





def inference_latent_diffusion(
    diffusion_checkpoint: str='latent_diffusion_checkpoint.pt',
    vae_checkpoint: str='vae_checkpoint.pt',
    num_time_steps: int=1000,
    latent_dim: int=4,
    ema_decay: float=0.999,
    num_samples: int=10
):

    vae = VAE(in_channels=3, latent_dim=latent_dim, base_channels=128).cuda()
    vae_ckpt = torch.load(vae_checkpoint)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    model = Unet(down_channels = (64, 128, 256, 512), 
                 up_channels = (512, 256, 128, 64),
                 input_dim=latent_dim).cuda()
    diff_ckpt = torch.load(diffusion_checkpoint)
    model.load_state_dict(diff_ckpt['model_state_dict'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(diff_ckpt['ema'])
    model.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda()
    scheduler.beta = scheduler.beta.cuda()
    
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    
    with torch.no_grad():
        model = ema.module.eval()
        z = torch.randn(1, latent_dim, 8, 8).cuda()
        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.tensor([t]).cuda()
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            z = ((1 / torch.sqrt(1 - scheduler.beta[t]))*z - (temp * model(z, t_tensor)))
            if t in times:
                decoded = vae.decode(z)
                images.append(decoded.cpu())
            noise = torch.randn_like(z)
            z = z + (noise * torch.sqrt(scheduler.beta[t]))
        
        t_tensor = torch.tensor([0]).cuda()
        temp = scheduler.beta[0]/( (torch.sqrt(1-scheduler.alpha[0]))*(torch.sqrt(1-scheduler.beta[0])) )
        x = (1/(torch.sqrt(1-scheduler.beta[0])))*z - (temp*model(z.cuda(),t_tensor).cpu())
        final_image = vae.decode(x)
        images.append(final_image.cpu())
    
    display_reverse(images)
    
    return images




def inference_monocular_depth(
    image: torch.Tensor,
    diffusion_checkpoint: str='ddpm_checkpoint_monocular_depth.pt',
    num_time_steps: int=1000,
    ema_decay: float=0.999
):

    checkpoint = torch.load(diffusion_checkpoint)
    model = Unet(input_dim=4,output_dim=1).cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    model.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda()
    scheduler.beta = scheduler.beta.cuda()
    
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []
    
    with torch.no_grad():
        model = ema.module.eval()
        rgb = image.cuda()                        
        depth = torch.randn(1, 1, 64, 64).cuda() 
        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.tensor([t]).cuda()
            model_in = torch.cat([rgb, depth], dim=1)
            temp = (scheduler.beta[t]/( (torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t])) ))
            depth = ((1 / torch.sqrt(1 - scheduler.beta[t])) * depth - (temp * model(model_in, t_tensor)))
            if t in times:
                images.append(depth.clone())
            noise = torch.randn_like(depth)
            depth = depth + (noise * torch.sqrt(scheduler.beta[t]))

        t_tensor = torch.tensor([0]).cuda()
        model_in = torch.cat([rgb, depth], dim=1)
        temp = scheduler.beta[0] / (torch.sqrt(1 - scheduler.alpha[0]) * torch.sqrt(1 - scheduler.beta[0]))
        depth = ((1 / torch.sqrt(1 - scheduler.beta[0])) * depth - temp * model(model_in, t_tensor))
        images.append(depth)

        display_reverse(images)
    
    return images