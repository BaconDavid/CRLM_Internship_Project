
import torch
from torch import tensor
from torch.utils.data import WeightedRandomSampler
from monai.transforms import Compose
from monai.data import ImageDataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import json
import math
import nibabel as nib
import numpy as np


#################################################################################
#                                Utils functions                                #
#################################################################################

#write a function to check the path, make it function wrapper

def path_check(func):
    def wrapper(*args,**kwargs):
        cfg = args[0]
        path = cfg.visual_im.visual_out_path
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Create the {path} directory')
        return func(*args,**kwargs)
    return wrapper




@path_check
def visual_input(cfg,im,label,im_name, percentage_image=1):
    """
    args:
        percentage_image: random show the percentage of image
        image_visual_path: path to save the image
    """
    # save the image
    if_show = np.random.choice([True, False], p=[percentage_image, 1 - percentage_image])
    batch_size = im.shape[0]
    if if_show:
        rows = math.ceil(batch_size/2)
        fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 5))

        for i in range(batch_size):
            col = i % 2
 
   
            ax = axes[col]
            ax.imshow(im[i, 0, cfg.visual_im.slice, :, :], cmap='gray')  # which slice to show
            im_name_ = im_name[i].split('/')[-1]# for linux
            ax.set_title(f'Label: {im_name_} {label[i]}')
            ax.set_title(f'Label: {im_name_} {label[i]}')
            
            ax.axis('off')  # close axias

    # layout
        plt.tight_layout()
        try:
            plt.savefig(cfg.visual_im.visual_out_path  + im_name_ + '.png')
        except:
            im_name_ = im_name[i].split('\\')[-1]#for windows
            plt.savefig(cfg.visual_im.visual_out_path + im_name_ + '.png')
        plt.close()  

def apply_window_to_volume(batched_volumes, window_center, window_width):
    """
    Apply windowing to a batch of 3D volumes.
    :param batched_volumes: The input batch of 3D volumes.
    :param window_center: The center of the window (window level).
    :param window_width: The width of the window.
    :return: Windowed batch of 3D volumes.
    """
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    windowed_batched_volumes = np.clip(batched_volumes, lower_bound, upper_bound)
    return windowed_batched_volumes





class Balanced_sampler(WeightedRandomSampler):
    def __init__(self,labels:tensor,num_class=3,*args,**kwargs) -> None:
        """
        args:
            labels: torch tensor 
        """
        labels = np.asarray(labels).astype(int)
        class_freq = [len(np.where(labels==i)[0]) for i in range(num_class)]
        weights = [1.0/class_freq[label] for label in labels]
        #print(len(class_freq),len(weights))
        super().__init__(weights=weights,num_samples=len(weights),replacement=True,*args,**kwargs)



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
    def store_results(self,df,metric_name):
        """
        args:
            df: dataframe that store the results
            metric_name: store type. eg: TP,FP,TN,FN
        
        """
        print("saving results")
        filename = self.result_path + self.type + '_' + metric_name + '.csv'
        if not os.path.isfile(filename):
            df.to_csv(filename, index=False, mode='w')
        else:
            df.to_csv(filename, index=False, mode='a', header=False)


    def _path_check(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f'Create the {self.result_path} directory')


