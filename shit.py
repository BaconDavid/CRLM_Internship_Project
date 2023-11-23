from torch import tensor
import torch
import nibabel as nib

from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd
import os
from monai.transforms import Compose

from monai.data import ImageDataset,DataLoader

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
    CenterSpatialCrop
    )
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DataFiles:
    def __init__(self,data_path,label_path,label_name) -> None:
        self.data_path = data_path
        self.label_path = label_path
        self.label_name = label_name

    def get_images(self):
        return [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path)]

    def get_labels(self):
        return pd.read_csv(self.label_path)[self.label_name].values.tolist()

    def Data_check(self):
        assert len(self.get_images()) == len(self.get_labels()) , 'The number of images and labels are not equal'

    def generate_data_dic(self):
        pass



class Image_Dataset(ImageDataset):
    def __init__(self,image_files,labels,transform_methods=None,data_aug=True,label_name=None,*args,**kwargs):

        if data_aug:
            transform = Compose(transform_methods)
        else:
            transform = None

        super().__init__(image_files=image_files,labels=labels,transform=transform,*args, **kwargs)

    def __getitem__(self,index,*args,**kwargs):
        output = super().__getitem__(index,*args,**kwargs)

        #print(f"this is image {output[0][index]}")
        #print(f"this is image shape {output[0].shape}")
        print('this is image and labels',self.image_files[index],self.labels[index])
        return output


class Data_Loader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers=0,*args,**kwargs):
        super().__init__(dataset=dataset,batch_size=batch_size,num_workers=num_workers,*args,**kwargs)
        self.args = args
        self.kwargs = kwargs
    
    def build_train_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True,*self.args,**self.kwargs)

    def build_vali_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
    def build_test_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
Data = DataFiles('../Data/CT_Phase/Resample_222/','../Data/CT_Phase/True_Label/Phase_label_all.csv','Phase')
images_lst = Data.get_images()
labels_lst = Data.get_labels()
Data.Data_check()


transform_param = {"transform_methods":[
                                    EnsureChannelFirst(),
                                    # Data augmentation
                                    RandZoom(prob = 0.5, min_zoom=1.0, max_zoom=1.2),
                                    RandRotate(range_z = 0.35, prob = 0.8),
                                    RandFlip(prob = 0.5),
                                    # To tensor
                                    ToTensor()
                                    ]}

dataset = Image_Dataset(image_files=images_lst,labels=labels_lst,transform_methods=transform_param['transform_methods'],data_aug=True,label_name=None)
dataloader = Data_Loader(dataset=dataset,batch_size=1,num_workers=0)

dataloader = dataloader.build_train_loader()

for i,(im,label) in enumerate(dataloader):
    print(im.shape)