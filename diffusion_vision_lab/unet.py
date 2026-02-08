
import torch
from torch import nn
import math
from typing import Tuple
from attention import SelfAttention, CrossAttention

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, use_attn=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gnorm1 = nn.GroupNorm(32, out_ch)
        self.gnorm2 = nn.GroupNorm(32, out_ch)
        self.relu  = nn.ReLU()
        self.attn = SelfAttention2D(out_ch) if use_attn else nn.Identity()
        
    def forward(self, x, t ):

        h = self.gnorm1(self.relu(self.conv1(x)))

        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[:, :, None, None]

        h = h + time_emb

        h = self.gnorm2(self.relu(self.conv2(h)))

        h = self.attn(h)

        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class SelfAttention2D(nn.Module):
    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x):

        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(-1, -2)
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        x = x.transpose(-1, -2).view((n, c, h, w))
        return self.conv_output(x) + residue_long

class Unet(nn.Module):

    def __init__(self, 
                 down_channels: Tuple[int, ...] =(64, 128, 256, 512, 1024), 
                 up_channels: Tuple[int, ...] = (1024, 512, 256, 128, 64), 
                 input_dim: int=3):
        super().__init__()
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        self.conv0 = nn.Conv2d(input_dim, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1],  time_emb_dim, use_attn=(i >= len(down_channels)-3)) 
                                    for i in range(len(down_channels)-1)])

        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True, use_attn=(i < 2)) 
                                  for i in range(len(up_channels)-1)])
        

        self.output = nn.Conv2d(up_channels[-1], input_dim, 1)

    def forward(self, x, timestep):

        t = self.time_mlp(timestep)

        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

