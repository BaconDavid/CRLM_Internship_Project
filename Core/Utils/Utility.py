
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
        if not os.path.exists(args[2]):
            os.makedirs(args[2])
            print(f'Create the {args[2]} directory')
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
        print("this is visual image shape",im.shape)
        plt.imshow(im[0,0,:,:,0], cmap='gray')
        plt.title(f'Label:{label}')
        plt.savefig(image_visual_path + f'Label_{label}{np.random.randint(0,100,1).item()}.png')



import nibabel as nib
import numpy as np

import numpy as np

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
        print(len(class_freq),len(weights))
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
        df.to_csv(self.result_path + self.type + '_' + metric_name+ '.csv',index=False,mode='a')
    
    def _path_check(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print(f'Create the {self.result_path} directory')


