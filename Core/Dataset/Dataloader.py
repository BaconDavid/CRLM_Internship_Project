from monai.transforms import Compose,SpatialPad
from monai.data import ImageDataset,DataLoader
import torch
from torch import tensor
from torch.utils.data import WeightedRandomSampler
import nibabel as nib
import numpy as np
import pandas as pd
import os




class DataFiles:
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 label_name: str) -> None:
        """
        Args:
            data_path: csv path to the data
            label_path: path to the label
            label_name: name of the label
        """
        self.data_path = data_path
        self.label_path = label_path
        self.label_name = label_name

    def get_images(self):
        filenames = self.get_data_path()
        return [os.path.join(self.data_path, filename) for filename in filenames]

    def get_labels(self):
        return pd.read_csv(self.label_path)[self.label_name].values.tolist()
    
    def get_data_path(self):
        return pd.read_csv(self.label_path)['data_path'].values.tolist()

    def Data_check(self):
        assert len(self.get_images()) == len(self.get_labels()) , 'The number of images and labels are not equal'

    def generate_data_dic(self):
        pass



class Image_Dataset(ImageDataset):
    def __init__(self,
                 image_files: list,
                 labels: list,
                 transform_methods=None,
                 data_aug: bool = True,
                 label_name: str = None,
                 padding_size: tuple=(256,256,64),
                 seg_files: bool = False,
                 seg_transform: list = None,
                 *args,**kwargs):
        """
        Args:
            image_files: list of image files
            labels: list of labels
            transform_methods: list of transform methods
            data_aug: bool, whether to do data augmentation
            label_name: name of the label
            padding_size: tuple, the size of padding. For models that require fixed size
        """
        if data_aug:
            transform = Compose(transform_methods)
        else:
            transform = None

        super().__init__(image_files=image_files,labels=labels,transform=transform,*args, **kwargs)
        self.padding_size = padding_size

    def __getitem__(self,index,*args,**kwargs):
        output = super().__getitem__(index,*args,**kwargs)
        im,label = output[0],output[1]
        img_name = self.image_files[index]
        #print(im.shape)
        #if self.padding_size:
        #    padder = SpatialPad(self.padding_size)
        #    im = padder(im)
        output = (im,label,img_name)
        return output


class Data_Loader(DataLoader):
    def __init__(self,dataset,batch_size,num_workers=0,*args,**kwargs):
        super().__init__(dataset=dataset,batch_size=batch_size,num_workers=num_workers,*args,**kwargs)
        self.args = args
        self.kwargs = kwargs
    
    def build_train_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.num_workers,drop_last=True,*self.args,**self.kwargs)

    def build_vali_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)
    
    def build_test_loader(self):
        return DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers,drop_last=False,*self.args,**self.kwargs)