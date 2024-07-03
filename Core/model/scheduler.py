import torch
from monai.optimizers import WarmupCosineSchedule

def build_scheduler(cfg,optimizer,**kwargs):
    if cfg.Scheduler.scheduler_name == 'WarmupCosineSchedule':
        return WarmupCosineSchedule(optimizer,
                                    cfg.Scheduler.WarmupCosineScheduler.warmup_steps,
                                    cfg.Scheduler.WarmupCosineScheduler.t_total,**kwargs)
    
    elif cfg.TRAIN.scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer,**kwargs)
    
class WeightDecayScheduler:
    def __init__(self, optimizer, init_weight_decay, final_weight_decay, total_steps):
        self.optimizer = optimizer
        self.init_weight_decay = init_weight_decay
        self.final_weight_decay = final_weight_decay
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        # Calculate the new weight decay value
        new_weight_decay = self.init_weight_decay - \
            (self.init_weight_decay - self.final_weight_decay) * \
            (self.current_step / self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_weight_decay

    @property
    def get_weight_decay(self):
        return self.optimizer.param_groups[0]['weight_decay']
