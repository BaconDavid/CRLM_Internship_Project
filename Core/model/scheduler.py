import torch
from monai.optimizers import WarmupCosineSchedule

def build_scheduler(cfg,optimizer,**kwargs):
    if cfg.Scheduler.scheduler_name == 'WarmupCosineSchedule':
        return WarmupCosineSchedule(optimizer,
                                    cfg.Scheduler.WarmupCosineScheduler.warmup_steps,
                                    cfg.Scheduler.WarmupCosineScheduler.t_total,**kwargs)
    
    elif cfg.TRAIN.scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,**kwargs)
