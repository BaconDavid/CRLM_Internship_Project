from  Core.model import build_model
from Core.model import loss,optimizer,scheduler
from Core.model.train import train_loop,Validation,test
from Core.Dataset.Dataloader import DataFiles,Image_Dataset,Data_Loader
from Core.Utils.Utility import DataFiles,path_check
from Core.Utils import args
from Core.Utils.Metrics import Metrics
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
from monai.data import ImageDataset,DataLoader

import os
import torch
import numpy as np
import datetime


def main(data_path,label_path,epochs,mode='train'):
    """
    args:
        epochs: number of epochs
        mode: train/vali or test
        data_path: path to the data folder

    """

    #args
    args = args.parse_args()

    #prepare dataset
    Data = DataFiles(data_path,label_path)
    images_lst = Data.get_images()
    labels_lst = Data.get_labels()
    Data.Data_check()

    dataset = Image_Dataset(image_files=images_lst,labels=labels_lst,transform_methods=None,data_aug=True,label_name=None)

    if mode == 'train':

        model = build_model('resnet10',3)
        
        dataloader = Data_Loader.build_train_loader(dataset=dataset,batch_size=1,num_workers=0) 
        
        #set metric
        metrics = Metrics()
        

        #set scheduler,optimizer parameters

        optimizer_param = {"lr":0.001}
        scheduler_param = {"step_size":10,"gamma":0.1}

        loss_fun = loss.build_loss()
        scheduler_fun = scheduler.build_scheduler(scheduler_param) 
        optimizer_param = {"lr":0.001}
        optimizer_fun = optimizer.build_optimizer(model.parameters(),optimizer_param)
        #transform methods
        transform_param = {"transform_methods":[
                                EnsureChannelFirst(),
                                # Data augmentation
                                RandZoom(prob = 0.5, min_zoom=1.0, max_zoom=1.2),
                                RandRotate(range_z = 0.35, prob = 0.8),
                                RandFlip(prob = 0.5),
                                # To tensor
                                ToTensor()
                                ]}



        for epoch in range(epochs):
            ave_loss,y_pred = train_loop(model,dataloader,epoch,'cpu',optimizer_fun,scheduler_fun,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)
            

    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)


if __name__ == "__main__":
    TIME_RUN = datetime.datetime.now().strftime("%Y-%m-%d-min %H:%M").replace(' ', '_').replace('-', '_').replace(':', '_')
    SAVE_PATH = args.model_save_path + TIME_RUN + '/'