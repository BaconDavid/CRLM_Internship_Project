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
            return SELoss(self.cfg).build_loss()
        else:
            raise ValueError('task should be classification or regression or selective')
        


class GamblerLoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    
    def __call__(self,model_output, targets,*args, **kwds): 
        return self.Gambler_loss(model_output, targets)
    


    def Gambler_loss(self,model_output, targets):
        
        outputs = torch.nn.functional.softmax(model_output, dim=1)
        
        outputs, reservation = outputs[:, :-1], outputs[:, -1]
        
        gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
        
        doubling_rate = (gain + reservation / self.cfg.LOSS.SelectiveLoss.GamblerLoss.reward).log()  # 假设reward为1
        # 计算损失，即负的增益率的平均值
        loss = -doubling_rate.mean()
        return loss
    


class RegressionLoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.Regression.loss == 'MSE':
            return nn.MSELoss()
        
class ClassificationLoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.ClassificationLoss.loss == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()


class SELoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
            return GamblerLoss(self.cfg)
        if self.cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
            return SelectiveLoss(nn.CrossEntropyLoss(reduction='none'),self.cfg.LOSS.SelectiveLoss.coverage,self.cfg.LOSS.SelectiveLoss.lm)



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
    
   