from Core.Utils.Models import build_model
from Core.Utils import args
from Core.Utils.Metrics import Metrics
from Core.Utils.Utility import SaveResults

from Core.Utils import Swin_Transformer_Classification

from Core.model import optimizer,scheduler
from Core.model.loss import Loss
from Core.model.train import train_loop
from Core.model.Validation import Validation_loop
from Core.Utils.Utility import Balanced_sampler
from Core.Utils import Plot_Loss


from Core.Dataset.Dataloader import DataFiles,Image_Dataset,Data_Loader

from torch.utils.data import WeightedRandomSampler
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
    CenterSpatialCrop,
    Resize,
    NormalizeIntensity
    )
from monai.data import ImageDataset,DataLoader

import os
import torch
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold


def main(data_path,label_path,save_path,epochs,num_class,mode='train'):
    """
    args:
        epochs: number of epochs
        mode: train/vali or test
        data_path: path to the data folder

    """



    #prepare dataset
    Data = DataFiles(data_path,label_path,'Phase')
    images_lst = sorted(Data.get_images())
    labels_lst = Data.get_labels()
    
    Data.Data_check()
    
    stratify_kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    





    if mode == 'train':
        #set model
        for fold,(train_index,vali_index) in enumerate(stratify_kfold.split(images_lst,labels_lst)):
            #print('fuck',train_index)
            train_images = [images_lst[i] for i in train_index]
            train_labels = [labels_lst[i] for i in train_index]
            vali_images = [images_lst[i] for i in vali_index]
            vali_labels = [labels_lst[i] for i in vali_index]

                #save results
            tr_results = SaveResults(save_path + f"fold{fold}/",'train')

            #weighted sampler
            sampler = Balanced_sampler(train_labels,num_class)

            #transform methods
            transform_param = {"transform_methods":[
                                    EnsureChannelFirst(),
                                    # Data augmentation
                                    RandZoom(prob = 0.5, min_zoom=1.0, max_zoom=1.2),
                                    RandRotate(range_z = 0.35, prob = 0.8),
                                    RandFlip(prob = 0.5),
                                    #Resize((256,256,128)),
                                    NormalizeIntensity(),
                                    # To tensor
                                    ToTensor()
                                    ]}
            model = build_model("resnet10",num_class)
            #model =  Swin_Transformer_Classification.SwinUNETR(in_channels=1, out_channels=2, img_size=(256, 256, 128))
            tr_dataset = Image_Dataset(image_files=train_images,labels=train_labels,transform_methods=transform_param['transform_methods'],data_aug=True,label_name=None)
            tr_dataloader = Data_Loader(dataset=tr_dataset,batch_size=1,num_workers=0,**{'sampler':sampler,"shuffle":False}).build_train_loader() 

            val_dataset = Image_Dataset(image_files=vali_images,labels=vali_labels,transform_methods=[EnsureChannelFirst(),Resize((256,256,128)),ToTensor()],data_aug=True,label_name=None)
            val_dataloader = Data_Loader(dataset=val_dataset,batch_size=1,num_workers=0).build_vali_loader() 
            #set scheduler,optimizer parameters

            optimizer_param = {"lr":0.005}
            scheduler_param = {"step_size":2000,"gamma":0.1}

            loss_fun = Loss().build_loss()
            
            #optimizer_param = {"lr":0.001}
            optimizer_fun = optimizer.build_optimizer(model.parameters(),**optimizer_param)
            scheduler_fun = scheduler.build_scheduler(optimizer_fun,**scheduler_param) 
            
      
            
            epoch_loss_values, train_loss_epoch_x_axis = [], []
            val_loss_values, val_loss_epoch_x_axis = [], []

            for epoch in range(epochs):
                train_loss_epoch_x_axis.append(epoch)
                val_loss_epoch_x_axis.append(epoch)

                ave_loss,y_pred = train_loop(model,tr_dataloader,epoch,args.device,num_class,optimizer_fun,scheduler_fun,loss_fun,visual_im=True,visual_out_path=save_path + f"fold{fold}/visual_input/")
                metrics = Metrics(num_class,y_pred,train_labels)
                AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()

                metrics.calculate_metrics()
                singel_metric = metrics.generate_metrics_df(epoch+1)

                tr_results.store_results(tr_results.df_results(four_rate_dic,AUC,accuracy,F1,ave_loss,epoch))
                tr_results.store_results(singel_metric)
                epoch_loss_values.append(ave_loss)
            
                
        ###########validation##############
                
                loss_fun = Loss().build_loss()

                #save results
                val_results = SaveResults(save_path + f"fold{fold}/",'vali')

                ave_loss,y_pred = Validation_loop(model,val_dataloader,args.device,num_class,loss_fun,visual_im=True,visual_out_path=save_path + f"fold{fold}/visual_input_vali/")
                print('this is fucking average loss',ave_loss)
                metrics = Metrics(num_class,y_pred,vali_labels)
                print(f'this is y_true_lst:{metrics.y_true_label},this is y_pred_list{metrics.y_pred_label}')
                AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()
                metrics.calculate_metrics()
                singel_metric = metrics.generate_metrics_df(epoch+1)
                print(singel_metric,666666)

                val_results.store_results(val_results.df_results(four_rate_dic,AUC,accuracy,F1,ave_loss,epoch+1))

                val_results.store_results(singel_metric)

                val_loss_values.append(ave_loss)
                #plot loss
                #Plot_Loss(train_loss_epoch_x_axis,epoch_loss_values,val_loss_epoch_x_axis,val_loss_values,tr_results.result_path,epoch+1)

            



    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)


if __name__ == "__main__":

    args = args.parse_args()
    
    TIME_RUN = datetime.datetime.now().strftime("%Y-%m-%d-min %H:%M").replace(' ', '_').replace('-', '_').replace(':', '_')
    SAVE_PATH = args.result_save_path + TIME_RUN + '/'
    NUM_CLASS = 3
    #args

    #set the cross validation

    main(args.data_path,args.label_path,SAVE_PATH,args.epochs,NUM_CLASS,mode=args.mode)
    main()
