import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import monai

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
from Core.Utils.Utility import Balanced_sampler
from Core.Utils import Plot_Loss

from Core.Config.config import get_cfg_defaults
from Core.Dataset.Dataloader import DataFiles,Image_Dataset,Data_Loader

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
    NormalizeIntensity
    )
from monai.data import ImageDataset,DataLoader

import os
import torch
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold


def main(cfg,mode='train'):
    """
    args:
        epochs: number of epochs
        mode: train/vali or test
        data_path: path to the data folder

    """



    #prepare data and label path
    train_data_path = cfg.DATA.Train_dir
    vali_data_path = cfg.DATA.Valid_dir
    test_data_path = cfg.DATA.Test_dir
    label_dir = cfg.LABEL.label_dir
    #label name in excel
    label_name = cfg.LABEL.label_name
    train_data = DataFiles(train_data_path,label_dir,label_name)
    images_lst = sorted(train_data.get_images())
    labels_lst = train_data.get_labels()
    train_data.Data_check()
    
    #split data
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
            tr_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.save_name + f"/fold{fold}/",'train')

            #transform methods
            transform_param_train = {"transform_methods":[
                                    EnsureChannelFirst(),
                                    # Data augmentation
                                    RandZoom(prob = 0.3, min_zoom=1.0, max_zoom=1.2),
                                    RandRotate(range_z = 0.3, prob = 0.5),
                                    RandFlip(prob = 0.3),
                                    Resize((256,256,32)),
                                    NormalizeIntensity(),
                                    # To tensor
                                    ToTensor()
                                    ]}
           
            transform_param_val = {"transform_methods":[EnsureChannelFirst(),
                                                       Resize((256,256,32)),
                                                       NormalizeIntensity(),
                                                       ToTensor()]}

            tr_dataset = Image_Dataset(image_files=train_images,labels=train_labels,transform_methods=transform_param_train['transform_methods'],data_aug=cfg.TRAIN.data_aug)
            val_dataset = Image_Dataset(image_files=vali_images,labels=vali_labels,transform_methods=transform_param_val['transform_methods'],data_aug=cfg.VALID.data_aug)
 
            
            if cfg.TRAIN.Debug:
                print(3333)
                tr_dataset_sub = Subset(tr_dataset,range(int(len(tr_dataset)*0.2)))
                val_dataset_sub = Subset(val_dataset,range(int(len(val_dataset)*0.2)))
                #print("this is the length of train and vali dataset",len(tr_subset),len(val_subset))

                #labels and images for subset
                train_labels = [tr_dataset[i][1] for i in range(len(tr_dataset_sub))]
                vali_labels = [val_dataset[i][1] for i in range(len(val_dataset_sub))]
                train_images = [tr_dataset[i][0] for i in range(len(tr_dataset_sub))]
                vali_images = [val_dataset[i][0] for i in range(len(val_dataset_sub))]
                
                #sampler
                sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)

                tr_dataset = tr_dataset_sub
                val_dataset = val_dataset_sub


                tr_dataloader = Data_Loader(dataset=tr_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,sampler=sampler,batch_size=cfg.TRAIN.batch_size).build_train_loader() 
                val_dataloader = Data_Loader(dataset=val_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,batch_size=cfg.VALID.batch_size).build_vali_loader()
                print(len(tr_dataloader),'shit')
            else:
                sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)
                tr_dataloader = Data_Loader(dataset=tr_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,sampler=sampler,batch_size=cfg.TRAIN.batch_size,shuffle=False).build_train_loader() 
                val_dataloader = Data_Loader(dataset=val_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,batch_size=cfg.VALID.batch_size).build_vali_loader()


            ## get train_labels and image_labels

            #set best metric
                
            best_metric = 10000000

            #set model
            model = build_model(cfg)
            model.to(cfg.SYSTEM.DEVICE)
            
            #set scheduler,optimizer parameters

            loss_fun = Loss().build_loss()
             #optimizer_param = {"lr":0.001}
            optimizer_fun = optimizer.build_optimizer(cfg,model.parameters())

            if cfg.TRAIN.scheduler:
                scheduler_param = {'step_size':2000,'gamma':0.1}
                scheduler_fun = scheduler.build_scheduler(optimizer_fun,**scheduler_param) 
            else:
                scheduler_fun = None

           
            
            epoch_loss_values, train_loss_epoch_x_axis = [], []
            val_loss_values, val_loss_epoch_x_axis = [], []
            
         

            for epoch in range(cfg.TRAIN.num_epochs):
                model.train()
                
                train_loss_epoch_x_axis.append(epoch+1)
                val_loss_epoch_x_axis.append(epoch+1)
                ave_loss,y_pred,y_true = train_loop(cfg,model,tr_dataloader,epoch,optimizer_fun,scheduler_fun,loss_fun)

                metrics = Metrics(cfg.MODEL.num_class,y_pred,y_true)
                AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()

                metrics.calculate_metrics()
                singel_metric = metrics.generate_metrics_df(epoch+1)

                #save loss and metrics
                tr_results.store_results(tr_results.df_results(four_rate_dic,AUC,accuracy,F1,ave_loss,epoch),'metrics')
                tr_results.store_results(singel_metric,'Four_rate')
                epoch_loss_values.append(ave_loss)
            
                
        ###########validation##############
                
                

                #save results
                val_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.save_name + f"/fold{fold}/",'vali')
                ###############




                #################
                model.eval()

                ave_loss,y_pred,y_true = Validation_loop(cfg,model,val_dataloader,loss_fun)
                print('this is fucking average loss',ave_loss)

                metrics = Metrics(cfg.MODEL.num_class,y_pred,y_true)
                print(f'this is y_true_lst:{metrics.y_true_label},this is y_pred_list{metrics.y_pred_label}')
                AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()
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
                                'model':model.state_dict(),
                                'optimizer':optimizer_fun.state_dict(),
                                'loss':loss_fun.state_dict(),
                                'arch': cfg.MODEL.name
                            }
                    save_checkpoint(cfg.SAVE.save_dir + cfg.SAVE.save_name+ f"/fold{fold}/",save_dict,f'best_metric_{epoch+1}.pth')
                    best_metric = ave_loss
                #plot loss
                #Plot_Loss(train_loss_epoch_x_axis,epoch_loss_values,val_loss_epoch_x_axis,val_loss_values,tr_results.result_path,epoch+1)

            



    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)



if __name__ == "__main__":
    """
    
    args = args.parse_args()
    
    TIME_RUN = datetime.datetime.now().strftime("%Y-%m-%d-min %H:%M").replace(' ', '_').replace('-', '_').replace(':', '_')
    SAVE_PATH = args.result_save_path + TIME_RUN + '/'
    NUM_CLASS = 3
    #args

    #set the cross validation

    # build model


    main(args.data_path,args.label_path,SAVE_PATH,args.epochs,NUM_CLASS,mode=args.mode)
    """
    args = args.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg.TRAIN.lr,type(cfg.TRAIN.lr))
    print('successfully load the config file !')
    main(cfg,mode=args.mode)
