import torch
from monai.optimizers import WarmupCosineSchedule

def build_scheduler(cfg,optimizer,**kwargs):
    if cfg.TRAIN.scheduler_name == 'WarmupCosineSchedule':
        return WarmupCosineSchedule(optimizer,cfg.warmup_steps,cfg.t_total,**kwargs)
    elif cfg.TRAIN.scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,**kwargs)
