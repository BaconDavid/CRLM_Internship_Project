
from torch import tensor
from torch.utils.data import WeightedRandomSampler
import torch
from monai.metrics import get_confusion_matrix,compute_roc_auc
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
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


def convert_hu_to_grayscale(hu_images, hu_min=15, hu_max=80):
    # 将HU值裁剪到指定范围
    hu_images = np.clip(hu_images, hu_min, hu_max)
    # 转换到灰度值（这里简单地将HU值的范围从[hu_min, hu_max]线性映射到[0, 255]）
    grayscale_images = (hu_images - hu_min) / (hu_max - hu_min) * 255.0
    # 返回转换后的图像
    return grayscale_images.astype(np.float32)


        

def plot(train_loss_epoch_x_axis, epoch_loss_values, val_loss_epoch_x_axis, val_loss_values, path, current_epoch):
    """
    Generate and save three types of loss plots (train loss, test loss, and combined) as PNG images.
    Additionally, save the loss data and x-axis values as numpy arrays.

    Parameters:
        train_loss_epoch_x_axis (list or array): X-axis values for the train loss plot (epochs).
        epoch_loss_values (list or array): Train loss values corresponding to each epoch.
        val_loss_epoch_x_axis (list or array): X-axis values for the test loss plot (epochs).
        val_loss_values (list or array): Test loss values corresponding to each epoch.
        path (str): Directory path where the plots and numpy arrays will be saved.
        current_epoch (int): The current epoch number for which the plots are being generated.
    """

    plt.plot(train_loss_epoch_x_axis,epoch_loss_values, label='Train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Train loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/Train_loss.png')
    plt.clf()

    plt.plot(val_loss_epoch_x_axis , val_loss_values, label='Test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'test loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/test_loss.png')
    plt.clf()


    plt.plot(train_loss_epoch_x_axis,epoch_loss_values, label='Train loss')
    plt.plot(val_loss_epoch_x_axis , val_loss_values, label='Test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Train and test loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/combi_loss.png')
    plt.clf()

    # also save them as a npy file
    np.save(path+'/train_loss.npy', epoch_loss_values)
    np.save(path+'/val_loss.npy', val_loss_values)
    np.save(path+'/train_loss_epoch_x_axis.npy', train_loss_epoch_x_axis)
    np.save(path+'/val_loss_epoch_x_axis.npy', val_loss_epoch_x_axis)

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

class Image_Dataset(ImageDataset):
    def __init__(self,image_files,labels,transform_methods=None,data_aug=True,label_name=None,*args,**kwargs):

        if data_aug:
            transform = Compose(transform_methods)
        else:
            transform = None

        super().__init__(image_files=image_files,labels=labels,transform=transform,*args, **kwargs)


class Metrics():
    def __init__(self,num_class=2,y_pred=None,y_true_label=None):
        """
        args:
            y_pred: list of predicted tensor
            y_true_label: list of true labels
        """
        self.num_class = num_class
        self.four_rate_dic = {str(i):{'tp':0,'fp':0,'tn':0,'fn':0} for i in range(num_class)}
        self.y_true_label = y_true_label
        self.y_pred_label = [torch.argmax(y_pre,dim=1).item() for y_pre in y_pred]
        self.y_pred_one_hot = torch.nn.functional.one_hot(torch.tensor(self.y_pred_label),num_classes=self.num_class)
        self.y_true_one_hot = torch.nn.functional.one_hot(torch.tensor(y_true_label),num_classes=self.num_class)
      
    def get_roc(self,average='macro'):
        return compute_roc_auc(self.y_pred_one_hot,self.y_true_one_hot,average)
        

    def get_four_rate(self) -> tensor:
        """
        args:
            y_pred: (B,C) one-hot tensor
            y_true: (B,C) one-hot tensor
        """
        confu_matrix = get_confusion_matrix(self.y_pred_one_hot,self.y_true_one_hot)
        #calculate tp,fp,tn,fn
        for i in range(self.num_class):
            self.four_rate_dic[str(i)]['tp'] += confu_matrix[:,i,0].sum()
            self.four_rate_dic[str(i)]['fp'] += confu_matrix[:,i,1].sum() 
            self.four_rate_dic[str(i)]['tn'] += confu_matrix[:,i,2].sum() 
            self.four_rate_dic[str(i)]['fn'] += confu_matrix[:,i,3].sum()
        return self.four_rate_dic
    
    def get_accuracy(self) -> float:
        """
        args:
            y_pred_label: list of predicted labels
            y_true_label: list of true labels
        """
        accuracy = accuracy_score(self.y_pred_label,self.y_true_label)
        return accuracy
    
    def get_f1_score(self,average='macro') -> float:
        return f1_score(self.y_pred_label,self.y_true_label,average=average)


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
