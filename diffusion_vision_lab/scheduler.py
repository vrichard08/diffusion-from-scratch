import torch
from torch import nn
from typing import Union


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
    


class ContinuousCosineScheduler(nn.Module):

    def __init__(self, s: float = 0.008):
        super().__init__()
        self.s = s

    def alpha_sigma(self, t: Union[int,float]):

        t = torch.tensor(t, dtype=torch.float32)
        t = (t + self.s) / (1 + self.s)
        alpha = torch.cos(0.5 * torch.pi * t)
        sigma = torch.sin(0.5 * torch.pi * t)

        return alpha.view(-1,1,1,1), sigma.view(-1,1,1,1)

    def log_snr(self, t: torch.Tensor):
        alpha, sigma = self.alpha_sigma(t)
        return torch.log(alpha**2 / sigma**2)

    def forward(self, t: torch.Tensor):
        return self.alpha_sigma(t)