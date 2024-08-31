import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
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
        if self.cfg.LOSS.ClassificationLoss.loss == 'FocalLoss':
            return FocalLoss(alpha=self.cfg.LOSS.ClassificationLoss.FocalLoss.alpha, 
                             gamma=self.cfg.LOSS.ClassificationLoss.FocalLoss.gamma, 
                             reduction='mean',
                             device=self.cfg.SYSTEM.DEVICE)


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
        print('gain',gain)
        reward = torch.tensor(self.cfg.LOSS.SelectiveLoss.GamblerLoss.reward,dtype=torch.float32,requires_grad=True,device=self.cfg.SYSTEM.DEVICE)
        doubling_rate = (gain + reservation / reward).log()  # 假设reward为1
        print(doubling_rate,'doubling_rate')
        # 计算损失，即负的增益率的平均值
        loss = -doubling_rate.mean()
        print('gambelr loss & reward',loss,reward)
        return loss,None
    
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

        #self.loss_func = CommonLoss(self.cfg).build_loss()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
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
        print('emprical_coverage',emprical_coverage)
        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        print('shape of loss',self.loss_func(prediction_out, target))
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device=self.cfg.SYSTEM.DEVICE)
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device=self.cfg.SYSTEM.DEVICE))**2
        penulty *= self.lm
        print('penulty',penulty)

        # compute aux loss
        common_loss = CommonLoss(self.cfg).build_loss()
        aux_loss = common_loss(aux_out, target)
        
        #aux_loss = torch.nn.CrossEntropyLoss()(aux_out, target)
        # loss information dict 
        loss_dict={}
        loss_dict['emprical_coverage'] = emprical_coverage.detach().cpu().item()
        loss_dict['emprical_loss'] = emprical_risk.detach().cpu().item()
        loss_dict['penulty'] = penulty.detach().cpu().item()
        loss_dict['aux_loss'] = aux_loss.detach().cpu().item()
        selective_loss = self.alpha*(emprical_risk + penulty) + (1-self.alpha)*aux_loss
        print(selection_out,loss_dict,'6666')

        return selective_loss, loss_dict
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean',device='cpu'):
        """
        :param alpha: Balancing factor, can be a scalar (for all classes) or a tensor (for class-specific weights)
        :param gamma: Modulating factor to focus on hard examples
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (int, float)):
            self.alpha = torch.tensor([alpha, 1 - alpha])  # Use specified alpha for class 1, and 1 - alpha for class 0
        else:
            self.alpha = alpha  # Use class-specific alpha if provided as tensor
        self.gamma = gamma
        self.alpha = self.alpha.to(device)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions from the model (batch_size, num_classes), expected to be logits
        :param targets: Ground truth labels (batch_size)
        """
        # Compute log softmax probabilities
        logpt = F.log_softmax(inputs, dim=-1)
        # Gather log probabilities for the true class labels
        targets = targets.view(-1, 1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        # Compute probabilities
        pt = logpt.exp()

        # Compute class weights
        at = self.alpha.gather(0, targets.view(-1))

        # Compute Focal Loss
        loss = -at * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

        
class CommonLoss:
    """
    Replace CELoss with other loss functions.
    """
    def __init__(self, cfg):
        self.cfg = cfg
    
    def build_loss(self):
        return self.common_loss_look_up()

    def common_loss_look_up(self):
        loss_look_tabel = {
            "CELoss": nn.CrossEntropyLoss(),
            "FocalLoss": FocalLoss(alpha=self.cfg.LOSS.CommonLoss.FocalLoss.alpha, 
                                gamma=self.cfg.LOSS.CommonLoss.FocalLoss.gamma, 
                                reduction='mean',
                                device=self.cfg.SYSTEM.DEVICE),
        }
        return loss_look_tabel[self.cfg.LOSS.CommonLoss.loss]