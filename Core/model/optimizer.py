import torch

def build_optimizer(**kwargs):
    return torch.optim.Adam(**kwargs)


