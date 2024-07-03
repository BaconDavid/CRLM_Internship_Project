from asyncio import tasks
import re
import sys
import os
from numpy import block
import torch
sys.path.append("..") # Adds higher directory to python modules path.
from Network import Resnet
from Network import swin_unetr,Swin_TS_Sparse,SelectiveNet
from typing import Any

#from monai.networks.nets import resnet10,ResNet,ResNetBlock,ResNetBottleneck,resnet18
from Network.Resnet import resnet10,resnet18
from Network import SACNN

from dropblock import DropBlock3D, LinearScheduler
from torch import Tensor, dropout
from typing import Union


class Model:
    def __init__(self,cfg) -> None:
        """
        cfg:config file
        """
        self.cfg = cfg

   
    def build_model(self,**kwargs):
        if self.cfg.MODEL.task == 'classification':
            return ClassificationModel(self.cfg).build_model(**kwargs)
        
        #elif self.cfg.MODEL.name.startswith('SwinTrans'):
        #    model = SwinTransformer(self.cfg).build_model(**kwargs)
            
        elif self.cfg.MODEL.task == 'selective':
            return SelectiveModel(self.cfg).build_model(**kwargs)

        else:
            raise NotImplementedError(f"model {self.cfg.MODEL.name} not implemented")
        return model

class ClassificationModel(Model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self):
        if self.cfg.MODEL.name.startswith('Resnet'):
            model = ResNet(self.cfg).build_model()
        elif self.cfg.MODEL.name.startswith('SwinTrans'):
            model = SwinTransformer(self.cfg).build_model()
        else:
            raise NotImplementedError(f"model {self.cfg.MODEL.name} not implemented")
        return model
    
class SelectiveModel(Model):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self,**kwargs):
        #check which selectivenet
        if self.cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
            return GamblerNet(self.cfg).build_model(**kwargs)
        
        elif self.cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
            return SelectiveNet(self.cfg).build_model(**kwargs)


class ResNet(Model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self,**kwargs):
        if self.cfg.MODEL.name == "Resnet10":
            return resnet10(n_input_channels=self.cfg.MODEL.num_in_channels,
                            num_classes=self.cfg.MODEL.num_class,
                            widen_factor=1,
                            no_max_pool=False,
                            drop_rate = self.cfg.MODEL.drop_out,
                            task = self.cfg.MODEL.task,
                            selectivenet = self.cfg.MODEL.selectivenet,
                            block_inplanes = list(self.cfg.MODEL.Resnet10.block_inplanes),
                            **kwargs)
        elif self.cfg.MODEL.name == "Resnet18":
            return resnet18(n_input_channels=self.cfg.MODEL.num_in_channels, 
                            num_classes=self.cfg.MODEL.num_class, 
                            widen_factor=1,
                            no_max_pool=False,
                            drop_rate = self.cfg.MODEL.drop_out,
                            task = self.cfg.MODEL.task,
                            selectivenet = self.cfg.MODEL.selectivenet,
                            **kwargs)
    
    #def __get_inplanes(self):
        #return [64,128,256,512]






class SwinTransformer(Model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self,**kwargs):
        if self.cfg.MODEL.name == "SwinTransformer":
            return swin_unetr.Swin3DTransformer(img_size=(64,256,256),
                                                in_channels=self.cfg.MODEL.num_in_channels, 
                                                num_class=self.cfg.MODEL.num_class,
                                                num_heads=[3, 6, 12, 24],
                                                out_channels=1,
                                                depths = self.cfg.MODEL.SwinTransformer.block_depth,
                                                 **kwargs)
        elif self.cfg.MODEL.name == "SwinTransformerSparse":
            return Swin_TS_Sparse.SwinSparseTransformer(in_channels=self.cfg.MODEL.num_in_channels,
                                                          num_classes=self.cfg.MODEL.num_class,
                                                          img_size=(64,256,256),
                                                          num_heads=[3, 6, 12, 24],
                                                          out_channels=1,
                                                          dropout = self.cfg.MODEL.drop_out,
                                                          **kwargs)
        else:
            raise NotImplementedError(f"model {self.cfg.MODEL.name} not implemented")
    



class SelectiveNet(Model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self,**kwargs):
        if self.cfg.MODEL.name.startswith('Resnet'):
            model = ResNet(self.cfg).build_model(**kwargs)# choose main body of the model
            return model
                
        else:
            raise NotImplementedError(f"model {self.cfg.MODEL.name} not implemented")

class GamblerNet(Model):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def build_model(self,**kwargs):
        if self.cfg.MODEL.name.startswith('Resnet'):
            if self.cfg.MODEL.name == "Resnet10":
                model = resnet10(n_input_channels=self.cfg.MODEL.num_in_channels,
                                num_classes=self.cfg.MODEL.num_class + 1,
                                widen_factor=1,
                                no_max_pool=False,
                                drop_rate=self.cfg.MODEL.drop_out,
                                task=self.cfg.MODEL.task,
                                block_inplanes = list(self.cfg.MODEL.Resnet10.block_inplanes),
                                **kwargs)
                
            elif self.cfg.MODEL.name == "Resnet18":
                model = resnet18(n_input_channels=self.cfg.MODEL.num_in_channels, 
                            num_classes=self.cfg.MODEL.num_class + 1, 
                            widen_factor=1,
                            no_max_pool=False,
                            drop_rate=self.cfg.MODEL.drop_out,
                            task=self.cfg.MODEL.task,
                            **kwargs)
        return model
    
      

