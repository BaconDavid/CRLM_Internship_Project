from monai.networks.nets import resnet10
from Utils import Swin_Transformer_Classification

def build_model(model_name, num_classes, input_channel=1,pretrained=False, **kwargs):
    if model_name == "resnet10":
        return resnet10(n_input_channels=input_channel, num_classes=2, widen_factor=1,**kwargs)
    elif model_name == "SwinTransformer":
        return Swin_Transformer_Classification.SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=num_classes, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, norm_layer=None, ape=False, patch_norm=True, use_checkpoint=False,**kwargs)
    else:
        raise NotImplementedError(f"model {model_name} not implemented")
    
