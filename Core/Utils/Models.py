import sys
import os
import torch
sys.path.append("..") # Adds higher directory to python modules path.
import Swin_Transformer_Classification,Swin_TS_Sparse
from typing import Any

from monai.networks.nets import resnet10,ResNet,ResNetBlock,ResNetBottleneck
from Source_Code import SACNN

from dropblock import DropBlock3D, LinearScheduler
from torch import Tensor, dropout
from typing import Union
def get_inplanes():
    return [64,128,256,512]



def build_model(cfg,**kwargs):


    if cfg.MODEL.name == "Resnet10":
        if cfg.MODEL.Drop_block:
            print(f'Model {cfg.MODEL.name} with dropblock!')
            return resnet10_Drop(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=False,drop_prob=0.9,block_size=5,**kwargs)
        print('load model',cfg.MODEL.name)
        return resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=False,**kwargs)
        
    elif cfg.MODEL.name == "SwingTransformer":
            print('load model',cfg.MODEL.name)
            if cfg.MODEL.pretrained:
                freezing_layers = cfg.MODEL.freeze_layers

                
                model = Swin_Transformer_Classification.Swintransformer(img_size=(64,256,256),
                                                                         in_channels=1, 
                                                                         num_classes=cfg.MODEL.num_class,
                                                                           num_heads=[3, 6, 12, 24],
                                                                           out_channels=1,
                                                                           feature_size=48,
                                                                           drop_rate=cfg.TRAIN.drop_out,
                                                                           **kwargs)
                model_pretrained = torch.load(cfg.MODEL.pretrained_path)
                model.load_from(weights=model_pretrained)
                model = freeze_layers(model,freezing_layers)
                return model
            else:
                return Swin_Transformer_Classification.Swintransformer(img_size=(64,256,256),in_channels=1, num_classes=cfg.MODEL.num_class, num_heads=[3, 6, 12, 24],out_channels=1,drop_rate=cfg.TRAIN.drop_out,use_v2=cfg.MODEL.v2,**kwargs)
    elif cfg.MODEL.name == "SACNN":
        return SACNN.resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=True,**kwargs)
    elif cfg.MODEL.name == 'SwingTransformerSparse':
        return Swin_TS_Sparse.SwinSparseTransformer(in_channels=1, num_classes=cfg.MODEL.num_class, img_size=(64,256,256),num_heads=[3, 6, 12, 24],out_channels=1,**kwargs)
    else:
        raise NotImplementedError(f"model {cfg.MODEL.name} not implemented")
    

def freeze_layers(model,freeze_layers):
    for layer in freeze_layers:
        for param_name,param_value in model.named_parameters():
            if layer in param_name:
                param_value.requires_grad = False
    return model



class ResNetCustom(ResNet):

    def __init__(self, block,layers, block_inplanes,drop_prob=0., block_size=5,*args,**kwargs):
        super().__init__(block,layers,block_inplanes,*args, **kwargs)
        self.dropblock = LinearScheduler(
            DropBlock3D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5000
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)
        print(x.shape,'this is x shape')
        x = self.avgpool(x)


        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x

def _resnet_drop(
    arch: str,
    block: Union[type[ResNetBlock], type[ResNetBottleneck]],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool,
    progress: bool,
    drop_prob: int,
    block_size: int,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNetCustom(block, layers, block_inplanes,drop_prob,block_size, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
            "Please ensure you pass the appropriate `shortcut_type` and `bias_downsample` args. as specified"
            "here: https://github.com/Tencent/MedicalNet/tree/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b#update20190730"
        )
    return model

def resnet10_Drop(pretrained: bool = False, progress: bool = True, drop_prob=0,block_size=5,**kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_drop("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, drop_prob,block_size,**kwargs)