from torch import tensor
import torch
import nibabel as nib

from torch.utils.data import WeightedRandomSampler

import numpy as np
import pandas as pd
import os
from monai.transforms import Compose

from monai.data import ImageDataset,DataLoader


class DataFiles:
    def __init__(self,data_path,label_path) -> None:
        self.data_path = data_path
        self.label_path = label_path

    def get_images(self):
        return [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path)]

    def get_labels(self,label_name):
        return pd.read_csv(self.label_path)[label_name].values.tolist()

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

        print(f"this is image {self.image_files[index]};This is label {self.labels[index]")

        return output


class Data_Loader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers=0,*args,**kwargs):
        super().__init__(dataset=dataset,batch_size=batch_size,num_workers=num_workers,*args,**kwargs)
    
    
    def build_train_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=True,*self.args,**self.kwargs)

    def build_vali_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
    def build_test_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
