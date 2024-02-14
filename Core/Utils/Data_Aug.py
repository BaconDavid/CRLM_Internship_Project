import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from Config.config import get_cfg_defaults

from monai.transforms import (
    EnsureChannelFirst,
    RandZoom,
    Compose,
    RandRotate,
    RandFlip,
    RandGaussianNoise,
    ToTensor,
    Resize,
    Rand3DElastic,
    RandSpatialCrop,
    ScaleIntensityRange,
    CenterSpatialCrop,
    Resize,
    NormalizeIntensity,
    ResizeWithPadOrCrop,
    SpatialPad,
    RandSpatialCrop
    )

def data_aug(cfg):
    if cfg.MODEL.name == 'Resnet10':
        
        data_aug_dict_train = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            'RandZoom':RandZoom(prob=0.3, min_zoom=1.0, max_zoom=1.2),
            'RandRotate':RandRotate(range_z=0.3,prob=0.5),
            'RandFlip':RandFlip(prob=0.3),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }

        data_aug_dict_vali = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }




    elif cfg.MODEL.name == 'SwingTransformer':
        data_aug_dict_train = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            #'Resize':Resize(cfg.Augmentation.Resize),
             'SpatialPad':SpatialPad(cfg.Augmentation.SpatialPad),
            # 'CenterSpatialCrop':CenterSpatialCrop(cfg.Augmentation.CenterSpatialCrop),
            'RandSpatialCrop':RandSpatialCrop((256,256,64),random_size=False,random_center=True),
            'RandZoom':RandZoom(prob=0.3, min_zoom=1.0, max_zoom=1.2),
            'RandRotate':RandRotate(range_z=0.3,prob=0.5),
            'RandFlip':RandFlip(prob=0.3),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }

        data_aug_dict_vali = {
            'EnsureChannelFirst':EnsureChannelFirst(),
            'Resize':Resize(cfg.Augmentation.Resize),
            'SpatialPad':SpatialPad(cfg.Augmentation.SpatialPad),
            'CenterSpatialCrop':CenterSpatialCrop(cfg.Augmentation.CenterSpatialCrop),
            'NormalizeIntensity':NormalizeIntensity(),
            'ToTensor':ToTensor(),
        }
    return list(data_aug_dict_train.values()),list(data_aug_dict_vali.values())


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    print(os.listdir('../Config/'))
    cfg.merge_from_file('../Config/SwingTransformer.yaml')
    print(cfg)
    print(data_aug(cfg))