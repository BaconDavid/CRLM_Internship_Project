import sys
import torch
sys.path.append("..") # Adds higher directory to python modules path.
from monai.networks.nets import resnet10
import Swin_Transformer_Classification,Swin_TS_Sparse
from Source_Code import SACNN

def build_model(cfg,**kwargs):


    if cfg.MODEL.name == "Resnet10":
        return resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,no_max_pool=False,**kwargs)
    elif cfg.MODEL.name == "SwingTransformer":
            if cfg.MODEL.pretrained:
                freezing_layers = cfg.MODEL.freeze_layers

                
                model = Swin_Transformer_Classification.Swintransformer(img_size=(64,256,256),
                                                                         in_channels=1, 
                                                                         num_classes=cfg.MODEL.num_class,
                                                                           num_heads=[3, 6, 12, 24],
                                                                           out_channels=1,
                                                                           feature_size=48,
                                                                           **kwargs)
                model_pretrained = torch.load(cfg.MODEL.pretrained_path)
                model.load_from(weights=model_pretrained)
                model = freeze_layers(model,freezing_layers)
                return model
            else:
                return Swin_Transformer_Classification.Swintransformer(img_size=(64,256,256),in_channels=1, num_classes=cfg.MODEL.num_class, num_heads=[3, 6, 12, 24],out_channels=1,**kwargs)
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