import torch

def build_optimizer(cfg,params,**kwargs):
    return torch.optim.Adam(params,lr=cfg.TRAIN.lr,weight_decay = cfg.MODEL.weight_decay,**kwargs)


