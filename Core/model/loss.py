import torch.nn as nn


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
        if self.cfg.LOSS.loss_name == 'CrossEntropy':
            return nn.CrossEntropyLoss()
    

