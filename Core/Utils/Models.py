from monai.networks.nets import resnet10
import Swin_Transformer_Classification

def build_model(cfg,**kwargs):
    if cfg.MODEL.name == "Resnet10":
        return resnet10(n_input_channels=cfg.MODEL.num_in_channels, num_classes=cfg.MODEL.num_class, widen_factor=1,**kwargs)
    elif cfg.MODEL.name == "SwinTransformer":
        return Swin_Transformer_Classification.SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=cfg.MODEL.num_class, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3, norm_layer=None, ape=False, patch_norm=True, use_checkpoint=False,**kwargs)
    else:
        raise NotImplementedError(f"model {cfg.MODEL.name} not implemented")
    
