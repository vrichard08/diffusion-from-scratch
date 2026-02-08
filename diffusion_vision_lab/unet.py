
import torch
from torch import nn
import math
from typing import Tuple

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
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        
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

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.query = nn.Conv2d(ch, ch, 1)
        self.key   = nn.Conv2d(ch, ch, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1)          
        k = self.key(x).view(B, C, -1)            
        v = self.value(x).view(B, C, -1)          
        
        attn = torch.bmm(q.permute(0, 2, 1), k)   
        attn = self.softmax(attn / (C ** 0.5))
        
        out = torch.bmm(v, attn.permute(0, 2, 1)) 
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x

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

