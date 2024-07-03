import torch

def build_optimizer(cfg,params,**kwargs):
    if cfg.Optimizer.name == 'Adam':
        return torch.optim.Adam(params,lr=cfg.Optimizer.lr,
                                weight_decay=cfg.Optimizer.weight_decay,
                                **kwargs)
    elif cfg.Optimizer.name == 'SGD':
        return torch.optim.SGD(params,lr=cfg.Optimizer.lr,
                               momentum=0.9,
                               weight_decay = cfg.Optimizer.weight_decay,
                               **kwargs)


