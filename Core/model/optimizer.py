import torch

def build_optimizer(params,**kwargs):
    return torch.optim.Adam(params,**kwargs)


