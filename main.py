import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import monai
from torch.utils.data import Subset
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
    NormalizeIntensity,
    ResizeWithPadOrCrop,
    SpatialPad
    )
from monai.data import ImageDataset,DataLoader

from Core.Utils.Models import build_model
from Core.Utils import args
from Core.Utils.Metrics import Metrics
from Core.Utils.Utility import SaveResults
from Core.model.checkpoint import save_checkpoint,load_checkpoint
from Core.Utils import Swin_Transformer_Classification
from Core.model import optimizer,scheduler
from Core.model.loss import Loss
from Core.model.train import train_loop
from Core.model.Validation import Validation_loop
from Core.Utils.Utility import Balanced_sampler, visual_input
from Core.Config.config import get_cfg_defaults
from Core.Dataset.Dataloader import DataFiles,Image_Dataset,Data_Loader

import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
from ema_pytorch import EMA

def main(cfg,mode='train'):
    """
    args:
        cfg: cfg configuration file
        mode: train/vali or test
    """
    data_path = cfg.DATA.Data_dir
    train_data_label = cfg.DATA.Train_dir
    vali_data_label = cfg.DATA.Valid_dir
    label_name = cfg.LABEL.label_name
    train_data = DataFiles(data_path,train_data_label,label_name)
    vali_data = DataFiles(data_path,vali_data_label,label_name)
    train_images = sorted(train_data.get_images())
    train_labels = train_data.get_labels()
    vali_images = sorted(vali_data.get_images())
    vali_labels = vali_data.get_labels()
    train_data.Data_check()
    vali_data.Data_check()
    
    if mode == 'train':

        #save results
        tr_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.fold +'/', 'train')
        padding = cfg.Preprocess.padding_size
        #transform methods

        transform_param_train = {"transform_methods":[
                                EnsureChannelFirst(),
                                # Data augmentation
                                RandZoom(prob = 0.3, min_zoom=1.0, max_zoom=1.2),
                                RandRotate(range_z = 0.3, prob = 0.5),
                                RandFlip(prob = 0.3),
                                Resize((256,256,-1)),
                                SpatialPad(padding),
                                CenterSpatialCrop(roi_size=(256,256,64)),
                                NormalizeIntensity(),
                                # To tensor
                                ToTensor()
                                ]}
        
        transform_param_val = {"transform_methods":[EnsureChannelFirst(),
                                                    Resize((256,256,-1)),
                                                    SpatialPad(padding),
                                                    CenterSpatialCrop((256,256,64)),
                                                    NormalizeIntensity(),
                                                    ToTensor()]}
      

        tr_dataset = Image_Dataset(image_files=train_images,labels=train_labels,transform_methods=transform_param_train['transform_methods'],data_aug=cfg.TRAIN.data_aug,padding_size=padding)
        val_dataset = Image_Dataset(image_files=vali_images,labels=vali_labels,transform_methods=transform_param_val['transform_methods'],data_aug=cfg.VALID.data_aug,padding_size=padding)

            
        if cfg.TRAIN.Debug:
            #how many data for subset
            tr_dataset_sub = Subset(tr_dataset,range(int(len(tr_dataset)*0.2)))
            val_dataset_sub = Subset(val_dataset,range(int(len(val_dataset)*0.2)))
            #labels and images for subset
            train_labels = [tr_dataset[i][1] for i in range(len(tr_dataset_sub))]
            
            #sampler
            if cfg.DATASET.WeightedRandomSampler:
                sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)
            else:
                sampler = None

            tr_dataset = tr_dataset_sub
            val_dataset = val_dataset_sub


            tr_dataloader = Data_Loader(dataset=tr_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,sampler=sampler,batch_size=cfg.TRAIN.batch_size).build_train_loader() 
            val_dataloader = Data_Loader(dataset=val_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,batch_size=cfg.VALID.batch_size).build_vali_loader()
            print(len(tr_dataloader),'shit')
        else:           
            #sampler
            if cfg.DATASET.WeightedRandomSampler:
                sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)
            else:
                sampler = None
            
            sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)
            tr_dataloader = Data_Loader(dataset=tr_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,sampler=sampler,batch_size=cfg.TRAIN.batch_size,shuffle=False).build_train_loader() 
            val_dataloader = Data_Loader(dataset=val_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,batch_size=cfg.VALID.batch_size).build_vali_loader()


            ## get train_labels and image_labels

            #set best metric
                
        best_metric = 10000000
        
        #set model
     
        model = build_model(cfg)
        model.to(cfg.SYSTEM.DEVICE)
        
        ## add exponential moving average
        ema = EMA(
        model,
        beta = 0.999,              # exponential moving average factor
        update_after_step = 50,    # only after this number of .update() calls will it start updating
        update_every = 10, 
        power =3/4 )
        #print(ema.step,"ema step!!")
        
        #set scheduler,optimizer parameters

        loss_fun = Loss().build_loss()
        optimizer_fun = optimizer.build_optimizer(cfg,model.parameters())

        if cfg.TRAIN.scheduler:
            scheduler_param = {'step_size':2000,'gamma':0.1}
            scheduler_fun = scheduler.build_scheduler(optimizer_fun,**scheduler_param) 
        else:
            scheduler_fun = None

        
        
        epoch_loss_values, train_loss_epoch_x_axis = [], []
        val_loss_values, val_loss_epoch_x_axis = [], []
        
         
        #visualize input
        if cfg.visual_im.visual_im:
            for im,label,im_name in tr_dataloader:
                #rotate and flip
                im = torch.rot90(im,k=3,dims=(2,3))
                im = torch.flip(im,[3])
                #permute to [B,C,D,H,W]
                im = im.permute(0,1,4,2,3)
                visual_input(im,label,im_name,cfg.visual_im.visual_out_path)

        for epoch in range(cfg.TRAIN.num_epochs):
            model.train()
            
            train_loss_epoch_x_axis.append(epoch+1)
            val_loss_epoch_x_axis.append(epoch+1)
            ave_loss,y_pred,y_true = train_loop(cfg,model,tr_dataloader,epoch,optimizer_fun,loss_fun,scheduler=scheduler_fun,ema=ema)

            metrics = Metrics(cfg.MODEL.num_class,y_pred,y_true)
            AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score('binary'),metrics.get_four_rate()

            metrics.calculate_metrics()
            singel_metric = metrics.generate_metrics_df(epoch+1)

            #save loss and metrics
            tr_results.store_results(tr_results.df_results(four_rate_dic,AUC,accuracy,F1,ave_loss,epoch),'metrics')
            tr_results.store_results(singel_metric,'Four_rate')
            epoch_loss_values.append(ave_loss)

    ###########validation##############
            #save results
            val_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.fold +'/','vali')
            ###############




            #################
            #model.eval()
            ema_model = ema.ema_model
            ema_model.eval()
            print(ema.step,'this is ema step')
            ave_loss,y_pred,y_true = Validation_loop(cfg,ema_model,val_dataloader,loss_fun)
            print('this is average loss',ave_loss)

            metrics = Metrics(cfg.MODEL.num_class,y_pred,y_true)
            print(f'this is y_true_lst:{metrics.y_true_label},this is y_pred_list{metrics.y_pred_label}')
            AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score('binary'),metrics.get_four_rate()
            metrics.calculate_metrics()
            singel_metric = metrics.generate_metrics_df(epoch+1)
            print(singel_metric,666666)
            
            #store metrics and loss
            val_results.store_results(val_results.df_results(four_rate_dic,AUC,accuracy,F1,ave_loss,epoch+1),'metrics')
            #store four rates
            val_results.store_results(singel_metric,'four rates')

            val_loss_values.append(ave_loss)

            #save best metric
            if ave_loss <= best_metric:
                save_dict = {
                            'epoch':epoch+1,
                            'model':ema_model.state_dict(),
                            'optimizer':optimizer_fun.state_dict(),
                            'loss':loss_fun.state_dict(),
                            'arch': cfg.MODEL.name
                        }
                save_checkpoint(cfg.SAVE.save_dir +  cfg.SAVE.fold,save_dict,f'best_metric_{epoch+1}.pth')
                best_metric = ave_loss
            #plot loss
            #Plot_Loss(train_loss_epoch_x_axis,epoch_loss_values,val_loss_epoch_x_axis,val_loss_values,tr_results.result_path,epoch+1)

        



    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)



if __name__ == "__main__":

    args = args.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg.TRAIN.lr,type(cfg.Preprocess.padding_size))
    print('successfully load the config file !')
    main(cfg,mode=args.mode)
