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

from Core.Config.config import get_cfg_defaults

from Core.Dataset.Dataloader import DataFiles,CreateImageDataset,CreateDataLoader

from Core.Utils.Metrics import ClassificationMetrics, Metrics,SelectiveMetrics
from Core.Utils.Utility import SaveResults, Balanced_sampler, visual_input
from Network import Swin_Transformer_Classification

from Core.Utils.Data_Aug import data_aug
from Core.Utils import args


from Core.model.Models import Model
from Core.model.optimizer import build_optimizer
from Core.model.scheduler import build_scheduler
from Core.model.loss import Loss
from Core.model.train import train_loop
from Core.model.Validation import Validation_loop
from Core.model.checkpoint import save_checkpoint



import numpy as np
import datetime
import random
from sklearn.model_selection import StratifiedKFold
from ema_pytorch import EMA

def main(cfg,mode='train'):
    """
    args:
        cfg: cfg configuration file
        mode: train/vali or test
    """

    ## set random seed
    torch.manual_seed(114514)
    torch.cuda.manual_seed_all(114514)
    np.random.seed(114514)
    random.seed(114514)
    torch.backends.cudnn.deterministic = True

    data_path = cfg.DATA.Data_dir
    mask_path = cfg.DATA.Data_mask_dir
    train_data_label = cfg.DATA.Train_file
    vali_data_label = cfg.DATA.Valid_file
    label_name = cfg.LABEL.label_name
    train_data = DataFiles(data_path,train_data_label,label_name)
    vali_data = DataFiles(data_path,vali_data_label,label_name)
    
    train_images = sorted(train_data.get_images())
    train_labels = train_data.get_labels()
  
    vali_images = sorted(vali_data.get_images())
    vali_labels = vali_data.get_labels()
    train_data.Data_check()
    vali_data.Data_check()

    y_pred_lst = []
    y_selection_lst = [] # only for selective net
    y_aux_lst = [] # only for selective net
    
    y_selection_tr_lst = [] # only for training selective net 
    y_pred_tr_lst = [] # only for training selective net 
    y_aux_tr_lst = []# only for training selective net

    y_true_tr_lst = [] # only for training selective net 
    y_true_lst = []
    
    if mode == 'train':
        tr_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.fold +'/', 'train') # save results
        transform_train,transform_val = data_aug(cfg) # data augmentation
        #whether to add mask as second channel
        if cfg.DATASET.mask:
            train_mask = DataFiles(mask_path,train_data_label,label_name)
            vali_mask = DataFiles(mask_path,vali_data_label,label_name)
            train_masks = train_mask.get_masks()
            vali_masks = vali_mask.get_masks()
            train_masks = sorted(train_masks)
            vali_masks = sorted(vali_masks)
            tr_dataset = CreateImageDataset(image_files=train_images,seg_files=train_masks,labels=train_labels,transform_methods=transform_train,data_aug=cfg.TRAIN.data_aug)
            val_dataset = CreateImageDataset(image_files=vali_images,seg_files=vali_masks,labels=vali_labels,transform_methods=transform_val,data_aug=cfg.VALID.data_aug)
        else:
            tr_dataset = CreateImageDataset(image_files=train_images,labels=train_labels,transform_methods=transform_train,data_aug=cfg.TRAIN.data_aug)
            val_dataset = CreateImageDataset(image_files=vali_images,labels=vali_labels,transform_methods=transform_val,data_aug=cfg.VALID.data_aug)

    #Debug model
        if cfg.TRAIN.Debug:
            tr_dataset_sub = Subset(tr_dataset,range(int(len(tr_dataset)*0.2))) #how many data for subset
            val_dataset_sub = Subset(val_dataset,range(int(len(val_dataset)*0.2)))
            train_labels = [tr_dataset[i][1] for i in range(len(tr_dataset_sub))] #labels and images for subset
            tr_dataset = tr_dataset_sub
            val_dataset = val_dataset_sub

        if cfg.DATASET.WeightedRandomSampler:
            sampler = Balanced_sampler(train_labels,num_class=cfg.MODEL.num_class)
        else:
            sampler = None

        tr_dataloader = CreateDataLoader(dataset=tr_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,sampler=sampler,batch_size=cfg.TRAIN.batch_size).build_train_loader() 
        val_dataloader = CreateDataLoader(dataset=val_dataset,num_workers=cfg.SYSTEM.NUM_WORKERS,batch_size=cfg.VALID.batch_size).build_vali_loader()

        #set best metric
        best_metric = 10000000
        
        #set model
        model = Model(cfg).build_model()
        model.to(cfg.SYSTEM.DEVICE)
        
        ## add exponential moving average
        ema = EMA(
        model,
        beta = 0.999,              # exponential moving average factor
        update_after_step = 50,    # only after this number of .update() calls will it start updating
        update_every = 10, 
        power =3/4 )
        
        #set scheduler,optimizer parameters
        loss_fun = Loss(cfg).build_loss()
        optimizer_fun = build_optimizer(cfg,model.parameters())
        if cfg.Scheduler.scheduler:
            scheduler_fun = build_scheduler(cfg,optimizer_fun) 
        else:
            scheduler_fun = None
        
        epoch_loss_values, train_loss_epoch_x_axis = [], []
        val_loss_values, val_loss_epoch_x_axis = [], []
    
        #visualize input
        if cfg.visual_im.visual_im:
            visual_input(cfg,tr_dataloader)

        for epoch in range(cfg.TRAIN.num_epochs):
            model.train()
            train_loss_epoch_x_axis.append(epoch+1)
            val_loss_epoch_x_axis.append(epoch+1)

            #which method to use
            if cfg.MODEL.task == 'classification':
                ave_loss,y_true,y_pred = train_loop(cfg,model,tr_dataloader,epoch,optimizer_fun,loss_fun,ema=ema,scheduler=scheduler_fun)
                #print(y_pred)
                metrics = ClassificationMetrics(y_true,y_pred,ave_loss,cfg.MODEL.num_class)
                metrics.calculate_metrics()
                metrics.get_four_rate()
                singel_metric = metrics.generate_metrics_df(epoch+1)
                four_rate_metric = metrics.generate_four_rate_df(epoch+1)

                #save loss and metrics
                tr_results.store_results(singel_metric,'metrics')
                tr_results.store_results(four_rate_metric,'four rates')

            elif cfg.MODEL.task == 'selective':
                if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                    ave_loss,y_true,y_pred = train_loop(cfg,model,tr_dataloader,epoch,optimizer_fun,loss_fun,ema=ema,scheduler=scheduler_fun)
                    metrics = SelectiveMetrics(y_true,
                                               y_pred,
                                               ave_loss,
                                               num_class=cfg.MODEL.num_class,
                                               coverage=[(i+1)/10 for i in range(10)],
                                               #coverage=[0.5],
                                               loss_type='GamblerLoss')
                    metrics.calculate_selected_metrics()
                    singel_metric = metrics.generate_metrics_df(epoch+1)
                    tr_results.store_results(singel_metric,'metrics')

                elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                    ave_loss,y_true,y_pred = train_loop(cfg,
                                                        model,
                                                        tr_dataloader,
                                                        epoch,
                                                        optimizer_fun,
                                                        loss_fun,
                                                        ema=ema,
                                                        scheduler=scheduler_fun)
                    metrics = SelectiveMetrics(y_true,
                                               y_pred,
                                               ave_loss,
                                               num_class=cfg.MODEL.num_class,
                                               coverage=[cfg.LOSS.SelectiveLoss.SelectiveNetLoss.coverage],
                                               loss_type='SelectiveLoss')
                    y_pred_tr_lst.append(metrics.y_pred)
                    y_selection_tr_lst.append(metrics.y_select)
                    y_true_tr_lst.append(metrics.y_true)

                    metrics.calculate_selected_metrics()
                    metrics.get_four_rate()
                    singel_metric = metrics.generate_metrics_df(epoch+1)
                    four_rate_metric = metrics.generate_four_rate_df(epoch+1)

                    tr_results.store_results(singel_metric,'metrics')
                    tr_results.store_results(four_rate_metric,'four rates')

            epoch_loss_values.append(ave_loss)

    ###########validation##############
            #save results
            val_results = SaveResults(cfg.SAVE.save_dir + cfg.SAVE.fold +'/','vali')


            ema_model = ema.ema_model
            ema_model.eval()
            ave_loss,y_pred,y_true = Validation_loop(cfg,ema_model,val_dataloader,loss_fun,epoch)

            
            print('this is average loss',ave_loss)
            # #save best metric
            
            # if (epoch) == 100:
            #     save_dict = {
            #                 'epoch':epoch+1,
            #                 'model':ema_model.state_dict(),
            #                 'optimizer':optimizer_fun.state_dict(),
            #                 'loss':loss_fun.state_dict(),
            #                 'arch': cfg.MODEL.name
            #             }
            #     save_checkpoint(cfg.SAVE.save_dir +  "weight/" + cfg.SAVE.fold,save_dict,f'best_metric_{epoch+1}.pth')
            #     best_metric = ave_loss
            
            # #save pred numpy array

            #save predict probability

            if cfg.MODEL.task == 'classification':
                metrics = ClassificationMetrics(y_true,y_pred,ave_loss,cfg.MODEL.num_class)
                #print(f'this is y_true_lst:{metrics.y_true},this is y_pred_list{metrics.y_pred_label}')
                #AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score('binary'),metrics.get_four_rate()
                metrics.calculate_metrics()
                metrics.get_four_rate()
                singel_metric = metrics.generate_metrics_df(epoch+1)
                four_rate_metric = metrics.generate_four_rate_df(epoch+1)

                y_pred_array = np.stack([y.detach().cpu().numpy() for y in y_pred],axis=0)
                y_pred_lst.append(y_pred_array)

                
                print(singel_metric)


                #store four rates
                val_results.store_results(singel_metric,'metrics')
                val_results.store_results(four_rate_metric,'four rates')

                val_loss_values.append(ave_loss)

            elif cfg.MODEL.task == 'selective':
                if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                    metrics = SelectiveMetrics(y_true,y_pred,ave_loss,num_class=cfg.MODEL.num_class,coverage=[(i+1)/10 for i in range(0,10)])
                    metrics.calculate_selected_metrics()
                    singel_metric = metrics.generate_metrics_df(epoch+1)
                    val_results.store_results(singel_metric,'metrics')
                    y_pred_array = np.stack([y.detach().cpu().numpy() for y in y_pred],axis=0)
                    y_pred_lst.append(y_pred_array)
                    val_loss_values.append(ave_loss)

                elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                    #print(y_true,'validation_y_true')
                    metrics = SelectiveMetrics(y_true,
                                               y_pred,
                                               ave_loss,
                                               num_class=cfg.MODEL.num_class,
                                               coverage=[cfg.LOSS.SelectiveLoss.SelectiveNetLoss.coverage],
                                               loss_type='SelectiveLoss')
                    
                    y_pred_lst.append(metrics.y_pred)
                    y_selection_lst.append(metrics.y_select)
                    y_true_lst.append(metrics.y_true)
                    
                    metrics.calculate_selected_metrics()
                    metrics.get_four_rate()
                    singel_metric = metrics.generate_metrics_df(epoch+1)
                    four_rate_metric = metrics.generate_four_rate_df(epoch+1)

                    val_results.store_results(singel_metric,'metrics')
                    val_results.store_results(four_rate_metric,'four rates')

  
        #stack y_pred_lst except the selective loss
        if cfg.MODEL.task == 'selective':
            if cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                y_pred_array = np.stack(y_pred_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,cfg.MODEL.num_class)
                y_selection_array = np.stack(y_selection_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,1) # for selection head in SENet
                y_true_array = np.stack(y_true_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1)

                y_pred_tr_array = np.stack(y_pred_tr_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,cfg.MODEL.num_class)
                y_selection_tr_array = np.stack(y_selection_tr_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,1) # for selection head in SENet
                y_true_tr_array  = np.stack(y_true_tr_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1)

                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_pred.npy',y_pred_array)
                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_selection.npy',y_selection_array)
                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_true.npy',y_true_array)

                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_pred_tr.npy',y_pred_tr_array)
                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_selection_tr.npy',y_selection_tr_array)
                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_true_tr.npy',y_true_tr_array)

            elif cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                y_pred_array = np.stack(y_pred_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,cfg.MODEL.num_class+1)
                np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_pred.npy',y_pred_array)
                
        elif cfg.MODEL.task == 'classification':
            y_pred_array = np.stack(y_pred_lst,axis=0).reshape(cfg.TRAIN.num_epochs,-1,cfg.MODEL.num_class)
            np.save(cfg.SAVE.save_dir + cfg.SAVE.fold + '/' + 'y_pred.npy',y_pred_array)


            
            
            
    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=args.visual_out_path)



if __name__ == "__main__":

    args = args.parse_args()
    train_file = 'train_cv_' + args.fold + '.csv'
    vali_file = 'val_cv_' + args.fold + '.csv'

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    #which fold 
    cfg.SAVE.fold = args.fold
    #set train file and vali file
    cfg.DATA.Train_file += train_file
    cfg.DATA.Valid_file += vali_file 
    cfg.visual_im.visual_out_path = os.path.join(cfg.visual_im.visual_out_path,args.exp_name,'Visual',cfg.SAVE.fold) + '//'
    #set experiment name
    cfg.SAVE.save_dir = os.path.join(cfg.SAVE.save_dir,args.exp_name) + '//'
    cfg.freeze()
    print('successfully load the config file !')

    main(cfg,mode=args.mode)
