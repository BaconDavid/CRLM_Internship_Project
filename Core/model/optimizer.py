import torch

def build_optimizer(cfg,params,**kwargs):
    if cfg.TRAIN.optimizer == 'Adam':
        return torch.optim.Adam(params,lr=cfg.TRAIN.lr,**kwargs)
    elif cfg.TRAIN.optimizer == 'SGD':
        return torch.optim.SGD(params,lr=cfg.TRAIN.lr,momentum=0.9,**kwargs)



