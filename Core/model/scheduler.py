import torch

def build_scheduler(optimizer,**kwargs):
    return torch.optim.lr_scheduler.StepLR(optimizer,**kwargs)