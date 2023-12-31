import sys
sys.path.append("..") # Adds higher directory to python modules path.
from monai.networks.nets import resnet10
import Swin_Transformer_Classification
from Source_Code import SACNN

def build_model(cfg,**kwargs):
    if cfg.MODEL.name == "Resnet10":
        return resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=True,**kwargs)
    elif cfg.MODEL.name == "SwingTransformer":
        return Swin_Transformer_Classification.SwinUNETR(img_size=256,in_channels=1, num_classes=cfg.MODEL.num_class, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],out_channels=1,**kwargs)
    elif cfg.MODEL.name == "SACNN":
        return SACNN.resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=True,SA=True,**kwargs)
    else:
        raise NotImplementedError(f"model {cfg.MODEL.name} not implemented")
    
