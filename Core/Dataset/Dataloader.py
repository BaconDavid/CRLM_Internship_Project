from torch import tensor
import torch
import nibabel as nib

from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd
import os
from monai.transforms import Compose

from monai.data import ImageDataset,DataLoader


class Image_Dataset(ImageDataset):
    def __init__(self,image_files,labels,transform_methods=None,data_aug=True,label_name=None,*args,**kwargs):
        """
        args:
            image_files: list of image files path
            labels: list of labels
            transform_methods: list of transform methods
            data_aug: True if data augmentation is used
            label_name: name of the label
        
        """
        if data_aug:
            transform = Compose(transform_methods)
        else:
            transform = None

        super().__init__(image_files=image_files,labels=labels,transform=transform,*args, **kwargs)

    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,index):
        image = self.image_files[index]
        label = self.labels[index]

        # get image array
        image = nib.load(image).get_fdata()
        # here to do windowing

        image = tensor(image)
        return image,label
    
class Data_Loader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers=0,*args,**kwargs):
        super().__init__(dataset=dataset,batch_size=batch_size,num_workers=num_workers,*args,**kwargs)
    
    
    def build_train_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True,*self.args,**self.kwargs)

    def build_vali_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
    def build_test_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
