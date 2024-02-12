import torch

def build_optimizer(cfg,params,**kwargs):
    return torch.optim.Adam(params,lr=cfg.TRAIN.lr,**kwargs)


