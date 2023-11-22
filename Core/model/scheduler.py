import torch

def build_scheduler(**kwargs):
    return torch.optim.lr_scheduler.StepLR(**kwargs)