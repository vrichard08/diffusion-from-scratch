import torch

def get_v_target(x, e, alpha_t):
    return torch.sqrt(alpha_t) * e - torch.sqrt(1 - alpha_t) * x