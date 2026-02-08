
import torch
from torch import nn
import numpy as np
import random
from timm.utils import ModelEmaV3 
from tqdm import tqdm
from unet import Unet
from vae import VAE
from dataset import CelebA
from scheduler import DDPM_Scheduler

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(batch_size: int=128,
          img_size: int=64,
          num_time_steps: int=1000,
          num_epochs: int=50,
          seed: int=-1,
          ema_decay: float=0.999,  
          lr=1e-4,
          checkpoint_path: str=None):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

    train_dataset = CelebA.load_transformed_dataset(img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda() 
    model = Unet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = torch.nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            t = torch.randint(0,num_time_steps,(batch_size,)).cuda()
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1)
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / len(train_loader)}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, 'ddpm_checkpoint.pt')




def train_vae(batch_size: int=128, img_size: int=64, num_epochs: int=20, 
              lr: float=1e-4, latent_dim: int=4, seed: int=42):

    set_seed(seed)
    
    train_dataset = CelebA.load_transformed_dataset(img_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=4
    )
    
    vae = VAE(in_channels=3, latent_dim=latent_dim, base_channels=128).cuda()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for x, _ in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}/{num_epochs}"):
            x = x.cuda()
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss, recon_loss, kld = VAE.vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kld = total_kld / len(train_loader)
        
        print(f'VAE Epoch {epoch+1} | Loss: {avg_loss:.4f} | '
              f'Recon: {avg_recon:.4f} | KLD: {avg_kld:.4f}')
        
        torch.save({
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, 'vae_checkpoint.pt')
    
    return vae



def train_latent_diffusion(
    batch_size: int=128, 
    img_size: int=64, 
    num_time_steps: int=1000, 
    num_epochs: int=50,
    seed: int=42, 
    lr: float=1e-4,
    ema_decay: float=0.999,
    latent_dim: int=4,
    vae_checkpoint: str='vae_checkpoint.pt'
):

    set_seed(seed)
    
    vae = VAE(in_channels=3, latent_dim=latent_dim, base_channels=128).cuda()
    checkpoint = torch.load(vae_checkpoint)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval() 


    for param in vae.parameters():
        param.requires_grad = False
    
    train_dataset = CelebA.load_transformed_dataset(img_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        drop_last=True, num_workers=4
    )
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.cuda()
    model = Unet(down_channels = (64, 128, 256, 512), 
                 up_channels = (512, 256, 128, 64),
                 input_dim=latent_dim).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    criterion = nn.MSELoss(reduction='mean')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for x, _ in tqdm(train_loader, desc=f"Diffusion Epoch {epoch+1}/{num_epochs}"):
            x = x.cuda()
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = vae.reparameterize(mu, logvar)
            t = torch.randint(0, num_time_steps, (batch_size,)).cuda()
            e = torch.randn_like(z, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1)
            x = (torch.sqrt(a) * z) + (torch.sqrt(1 - a) * e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        
        avg_loss = total_loss / len(train_loader)
        print(f'Diffusion Epoch {epoch+1} | Loss: {avg_loss:.5f}')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema': ema.state_dict(),
            'epoch': epoch
        }, 'latent_diffusion_checkpoint.pt')
    
    return model, vae

