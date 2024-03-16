
from collections import OrderedDict
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
def visual_input(cfg,data_loader, percentage_image=1):
    """
    args:
        percentage_image: random show the percentage of image
        image_visual_path: path to save the image
    """
    # get image,label,im_name
   
    for data in data_loader:
        if cfg.DATASET.mask:
            im,label,im_name,mask = data
        else:
            im,label,im_name = data
        batch_size = im.shape[0]
        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)
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


class Evaluator(object):
    def __init__(self, prediction_out, t, selection_out=None, selection_threshold:float=0.5):
        """
        Args:
            prediction_out (B, #class):
            t (B):  
            selection_out (B, 1)
        """
        assert 0<=selection_threshold<=1.0

        self.prediction_result = prediction_out.argmax(dim=1) # (B)
        self.t = t.detach() # (B)
        if selection_out is not None:
            condition = (selection_out >= selection_threshold)
            self.selection_result = torch.where(condition, torch.ones_like(selection_out), torch.zeros_like(selection_out)).view(-1) # (B)
        else:
            self.selection_result = None

    def __call__(self):
        """
        add 'accuracy (Acc)', 'precision (Pre)', 'recall (Rec)' to metric_dict. 
        if selection_out is not None, 'rejection rate (RR)' and 'precision of rejection (PR)' are added.
        
        Args:
            metric_dict
        """
        eval_dict = OrderedDict()

        if self.selection_result is None:
            # add evaluation for classification
            eval_dict_cls = self._evaluate_multi_classification(self.prediction_result, self.t)
            eval_dict.update(eval_dict_cls)

        else:
            # add evaluation for classification
            eval_dict_cls = self._evaluate_multi_classification_with_rejection(self.prediction_result, self.t, self.selection_result)
            eval_dict.update(eval_dict_cls)
            # add evaluation for rejection 
            eval_dict_rjc = self._evaluate_rejection(self.prediction_result, self.t, self.selection_result)
            eval_dict.update(eval_dict_rjc)

        return eval_dict


    def _evaluate_multi_classification(self, h:torch.tensor, t:torch.tensor):
        """
        evaluate result of multi classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1
            t (B): labels which indicates true label form 0 to #class-1
        Return:
            OrderedDict: accuracy
        """
        assert h.size(0) == t.size(0) > 0
        assert len(h.size()) == len(t.size()) == 1

        t = float(torch.where(h==t, torch.ones_like(h), torch.zeros_like(h)).sum())
        f = float(torch.where(h!=t, torch.ones_like(h), torch.zeros_like(h)).sum())

        # raw accuracy
        acc = float(t/(t+f+1e-12))
        return OrderedDict({'accuracy':acc})

    def _evaluate_multi_classification_with_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of multi classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1
            t (B): labels which indicates true label form 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: 'acc'/'raw acc'
        """
        assert h.size(0) == t.size(0) == r_binary.size(0)> 0
        assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

        # raw accuracy
        eval_dict = self._evaluate_multi_classification(h, t)
        eval_dict['raw accuracy'] = eval_dict['accuracy']
        del eval_dict['accuracy']

        h_rjc = torch.masked_select(h, r_binary.bool())
        t_rjc = torch.masked_select(t, r_binary.bool())

        t = float(torch.where(h_rjc==t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        f = float(torch.where(h_rjc!=t_rjc, torch.ones_like(h_rjc), torch.zeros_like(h_rjc)).sum())
        # accuracy
        acc = float(t/(t+f+1e-12))
        eval_dict['accuracy'] = acc

        return eval_dict


    def _evaluate_binary_classification(self, h:torch.tensor, t_binary:torch.tensor):
        """
        evaluate result of binary classification. 

        Args:
            h (B): binary prediction which indicates 'positive:1' and 'negative:0'
            t_binary (B): labels which indicates 'true1:' and 'false:0'
        Return:
            OrderedDict: accuracy, precision, recall
        """
        assert h.size(0) == t_binary.size(0) > 0
        assert len(h.size()) == len(t_binary.size()) == 1

        # conditions (true,false,positive,negative)
        condition_true  = (h==t_binary)
        condition_false = (h!=t_binary)
        condition_pos = (h==torch.ones_like(h))
        condition_neg = (h==torch.zeros_like(h))

        # TP, TN, FP, FN
        true_pos = torch.where(condition_true and condition_pos, torch.ones_like(h), torch.zeros_like(h))
        true_neg = torch.where(condition_true and condition_neg, torch.ones_like(h), torch.zeros_like(h))
        false_pos = torch.where(condition_false and condition_pos, torch.ones_like(h), torch.zeros_like(h))
        false_neg = torch.where(condition_false and condition_neg, torch.ones_like(h), torch.zeros_like(h))

        assert (true_pos + true_neg + false_pos + false_neg)==torch.ones_like(true_pos)

        tp = float(true_pos.sum())
        tn = float(true_neg.sum())
        fp = float(false_pos.sum())
        fn = float(false_neg.sum())

        # accuracy, precision, recall
        acc = float((tp+tn)/(tp+tn+fp+fn+1e-12))
        pre = float(tp/(tp+fp+1e-12))
        rec = float(tp/(tp+fn+1e-12))

        return OrderedDict({'accuracy':acc, 'precision':pre, 'recall':rec})


    def _evaluate_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of binary classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1 
            t (B): labels which indicates true class index from 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: rejection_rate, rejection_precision
        """
        assert h.size(0) == t.size(0) == r_binary.size(0)> 0
        assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

        # conditions (true,false,positive,negative)
        condition_true  = (h==t)
        condition_false = (h!=t)
        
        condition_acc = (r_binary==torch.ones_like(r_binary))
        condition_rjc = (r_binary==torch.zeros_like(r_binary))

        # TP, TN, FP, FN
        ta = float(torch.where(condition_true & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        tr = float(torch.where(condition_true & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fa = float(torch.where(condition_false & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fr = float(torch.where(condition_false & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())

        # accuracy, precision, recall
        rejection_rate = float((tr+fr)/(ta+tr+fa+fr+1e-12))
        rejection_pre  = float(tr/(tr+fr+1e-12))

        return OrderedDict({'rejection rate':rejection_rate, 'rejection precision':rejection_pre}) 