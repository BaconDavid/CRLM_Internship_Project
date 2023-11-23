
from torch import tensor
from torch.utils.data import WeightedRandomSampler
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from monai.transforms import Compose
from monai.data import ImageDataset
import json
#################################################################################
#                                Utils functions                                #
#################################################################################

#write a function to check the path, make it function wrapper

def path_check(func):
    def wrapper(*args,**kwargs):
        if not os.path.exists(args[0]):
            os.makedirs(args[0])
            print(f'Create the {args[0]} directory')
        return func(*args,**kwargs)
    return wrapper




@path_check
def visual_input(im, label,image_visual_path, percentage_image=0.5):
    """
    args:
        percentage_image: random show the percentage of image
        image_visual_path: path to save the image
    """
    # save the image
    if_show = np.random.choice([True, False], p=[percentage_image, 1 - percentage_image])

    if if_show:
        plt.imshow(im[:, :, 30], cmap='gray')
        plt.title(f'Label:{label}')
        plt.savefig(image_visual_path + f'Label_{label}.png')




        








class Balanced_sampler(WeightedRandomSampler):
    def __init__(self,labels:tensor,num_class=3,*args,**kwargs) -> None:
        """
        args:
            labels: torch tensor 
        """
        labels = np.asarray(labels).astype(int)
        class_freq = [len(np.where(labels==i)[0]) for i in range(num_class)]
        weights = [1.0/class_freq[label] for label in labels]
        print(len(class_freq),len(weights))
        super().__init__(weights=weights,num_samples=len(weights),replacement=True,*args,**kwargs)

class Parameters:
    def __init__(self) -> None:
        self._cross_val_par = None
        self._model_par = None
        self._loss_par = None
        self._model_info = None


    @property
    def cross_val_par(self):
        return self._cross_val_par
    
    @property
    def model_par(self):
        return self._model_par
    
    @property
    def loss_par(self):
        pass

    @property
    def model_info(self):
        return self._model_info

    @loss_par.setter
    def loss_par(self,value):
        self._loss_par = value
    
    @cross_val_par.setter
    def cross_val_par(self,value):
        self._cross_val_par = value

    @model_par.setter
    def model_par(self,value):
        self._model_par = value
    
    @model_info.setter
    def model_info(self,value):
        self._model_info = value

    @loss_par.getter
    def loss_par(self):
        return self._loss_par
    
    @cross_val_par.getter
    def cross_val_par(self):
        return self._cross_val_par
    
    @model_par.getter
    def model_par(self):
        return self._model_par
    
    @model_info.getter
    def model_info(self):
        return self._model_info



    def _aggregate_par(self):
        self.par = {**self._cross_val_par,**self._model_info,**self.model_par,**self._loss_par}
        self.par = {key: str(value) for key, value in self.par.items()}
        return self.par

    def write_model_info(self,out_path):
        #json dump
        model_info = self._aggregate_par()
        with open(out_path,'w') as f:
            #write txt
            for key,value in model_info.items():
                f.write(f'{key}:{value}\n')


class SaveResults:
    def __init__(self,result_path,type='train') -> None:
        """
        args:
            epoch: int
        """
        self.result_path = result_path
        self.type = type
        self._path_check()

    
    def df_results(self,four_rate_dic,auc,accuracy,f1,loss,epoch):
        #each epoch save the results
          # 使epoch从1开始编号，而不是0
        epoch_data = {'epoch':  epoch +1}

    # 遍历每个类别的统计数据
        for class_id, metrics in four_rate_dic.items():
            # 为每个度量指标创建一个键
            for metric_name, metric_value in metrics.items():
                # 创建一个新的键，格式为"class_x_metric"
                key = f'class_{class_id}_{metric_name}'
                # 将该度量的值分配给这个键
                epoch_data[key] = metric_value.item()  # 转换tensor为Python标量
        
        df = pd.DataFrame([epoch_data])      
        df['accuracy'] = accuracy
        df['roc_auc'] = auc
        df['f1_score'] = f1
        df['loss'] = loss

        return df
    def store_results(self,df):
        df.to_csv(self.result_path + self.type +'.csv',index=False,mode='a')
    
    def _path_check(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f'Create the {self.result_path} directory')


class VisualInput:
    def __init__(self,image_visual_path) -> None:
        self.image_visual_path = image_visual_path

    def visule_image(self,im,label,percentage_image=0.5):
        """
        args:
            percentage_image: random show the percentage of image
        """
        #save the image
        if_show = np.random.choice([True, False], p=[percentage_image, 1 - percentage_image])

        if if_show:
            plt.imshow(im[:,:,30],cmap='gray')
            plt.title(f'Label:{label}')
            plt.savefig(self.image_visual_path + f'Label_{label}.png')


    def show_image_label(self):
        print("Image:",self.im,"Label:",self.label)
