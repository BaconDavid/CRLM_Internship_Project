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
            return nn.CrossEntropyLoss(reduction='mean')


class SELoss(Loss):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    def build_loss(self):
        if self.cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
            return GamblerLoss(self.cfg)
        if self.cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
            return SelectiveLoss(self.cfg)


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
    
class SelectiveLoss(Loss):
    def __init__(self,cfg):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B). 
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32. 
        """
        super().__init__(cfg)

        assert 0.0 < self.cfg.LOSS.SelectiveLoss.SelectiveNetLoss.coverage <= 1.0
        assert 0.0 < self.cfg.LOSS.SelectiveLoss.SelectiveNetLoss.lm

        self.loss_func = nn.CrossEntropyLoss(reduce='none')
        self.coverage = self.cfg.LOSS.SelectiveLoss.SelectiveNetLoss.coverage
        self.lm = self.cfg.LOSS.SelectiveLoss.SelectiveNetLoss.lm
        self.alpha = self.cfg.MODEL.SelectiveNet.alpha # combine coefficient of selective loss and aux loss

    def __call__(self, prediction_out, selection_out, aux_out,target):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean() 
        print(selection_out.shape,'shape of selectionout')
        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device=self.cfg.SYSTEM.DEVICE)
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=self.cfg.SYSTEM.DEVICE))**2
        penulty *= self.lm

        # compute aux loss
        aux_loss = torch.nn.CrossEntropyLoss()(aux_out, target)
        # loss information dict 
        loss_dict={}
        loss_dict['emprical_coverage'] = emprical_coverage.detach().cpu().item()
        loss_dict['emprical_risk'] = emprical_risk.detach().cpu().item()
        loss_dict['penulty'] = penulty.detach().cpu().item()
        loss_dict['aux_loss'] = aux_loss.detach().cpu().item()
        selective_loss = self.alpha*(emprical_risk + penulty) + (1-self.alpha)*aux_loss

        return selective_loss, loss_dict
    
    
class CommonLoss:
    """
    Replace CELoss with other loss functions.
    """
    def __init__(self, cfg):
        self.cfg = cfg
    
    def build_loss(self):
        return loss_look_up(self.cfg)

def loss_look_up(cfg):
    loss_look_tabel = {
        "CELoss": nn.CrossEntropyLoss(),

    }
    pass