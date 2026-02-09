
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x + residual


class Encoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=4, base_channels=128):
        super().__init__()
        
        # downsampling: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1)  # 64->32
        )
        
        self.down2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)  # 32->16
        )
        
        self.down3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1)  # 16->8
        )
        
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4)
        
        # Output to mean and logvar
        self.conv_mu = nn.Conv2d(base_channels * 4, latent_dim, 3, padding=1)
        self.conv_logvar = nn.Conv2d(base_channels * 4, latent_dim, 3, padding=1)
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, out_channels=3, base_channels=128):
        super().__init__()
        
        self.init_conv = nn.Conv2d(latent_dim, base_channels * 4, 3, padding=1)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4)
        
        # upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),  # 8->16
            ResidualBlock(base_channels * 2, base_channels * 2)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),  # 16->32
            ResidualBlock(base_channels, base_channels)
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),  # 32->64
            ResidualBlock(base_channels, base_channels)
        )
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, z):
        x = self.init_conv(z)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final_conv(x)
        return torch.tanh(x)  # [-1, 1]


class VAE(nn.Module):

    def __init__(self, in_channels: int=3, latent_dim: int=4, base_channels: int=128):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, in_channels, base_channels)
        self.latent_dim = latent_dim
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    @staticmethod
    def vae_loss(recon_x, x, mu, logvar, beta=0.7):
        """VAE loss: reconstruction + KL divergence"""

        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld
        
        return total_loss, recon_loss, kld
