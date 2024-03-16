import torch.nn as nn
import torch

class Loss:
    def __init__(self,cfg):
        """
        args:
            args only have one loss function
        """
        self.cfg = cfg


    def build_loss(self):
        if self.cfg.MODEL.task == 'classification':
            return ClassificationLoss(self.cfg).build_loss()
        elif self.cfg.MODEL.task == 'regression':
            return RegressionLoss(self.cfg).build_loss()
        elif self.cfg.MODEL.task == 'selective':
            return SelectiveLoss(self.cfg).build_loss()
    
class RegressionLoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.loss_name == 'MSE':
            return nn.MSELoss()
        
class ClassificationLoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.loss_name == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif self.cfg.LOSS.loss_name == 'SelectiveLoss':
            return SelectiveLoss(nn.CrossEntropyLoss(),coverage=self.cfg.LOSS.coverage,lm=self.cfg.LOSS.lm)



class SelectiveLoss(nn.Module):
    def __init__(self, loss_func, coverage:float, lm:float=32.0):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B). 
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32. 
        """
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.loss_func = loss_func
        self.coverage = coverage
        self.lm = lm

    def forward(self, prediction_out, selection_out, target):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean() 

        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        penulty *= self.lm

        selective_loss = emprical_risk + penulty

        # loss information dict 
        loss_dict={}
        loss_dict['emprical_coverage'] = emprical_coverage.detach().cpu().item()
        loss_dict['emprical_risk'] = emprical_risk.detach().cpu().item()
        loss_dict['penulty'] = penulty.detach().cpu().item()

        return selective_loss, loss_dict
    
    def build_loss(self):
        return self