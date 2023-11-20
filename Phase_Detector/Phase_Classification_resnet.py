import os
import pandas as pd
import nibabel as nib
import datetime
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import label_binarize
import einops # for ViT
from torch import tensor

from torchsummary import summary
import monai
from monai.metrics import get_confusion_matrix,compute_roc_auc
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
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
import Utility
import argparse
import sys
import datetime

#################################################################################
#                                Data Loader                                #
#################################################################################


TIME_RUN = datetime.datetime.now().strftime("%Y-%m-%d-min %H:%M").replace(' ', '_').replace('-', '_').replace(':', '_')
DATA_PATH = '/data/scratch/r098986/CT_Phase/Data/Resample_Data_all/Resample_222_Data/'
LABEL_PATH = '/data/scratch/r098986/CT_Phase/Data/True_Label/Phase_label_all.csv'
SAVE_PATH = '/data/scratch/r098986/CT_Phase/Data/Results/'+TIME_RUN+"/"

#check pin,cuda
print(f"torch.cuda.is_available():{torch.cuda.is_available()}")
print(f"torch.cuda.device_count():{torch.cuda.device_count()}")
# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
# set the model
model = monai.networks.nets.resnet10(n_input_channels=1, num_classes=2, widen_factor=1).to(device)
sigmiod = torch.nn.Sigmoid()

#set the dataset and dataloader
data_files = Utility.DataFiles(DATA_PATH,label_path=LABEL_PATH)
image_files,labels = data_files.get_images(),data_files.get_labels('Phase')

#sort the image files and labels
image_files = sorted(image_files)
#print(f"this is imagefiles and labels",image_files[0:10],labels[0:10])

######check data########
check_ds = Utility.Image_Dataset(image_files,labels,transform_methods=[])
check_loader = DataLoader(dataset=check_ds,batch_size=1,num_workers=0,pin_memory=True)
im, label = monai.utils.misc.first(check_loader)

im = nib.load(image_files[0])
im = im.get_fdata()
print('im.shape: ', im.shape)
#################################################################################



#################################################################################
#                                Train model                                 #
#################################################################################
# set model info and parameters
Tr_Pars = Utility.Parameters()
Tr_Pars.model_par = {'lr':0.005,'max_epchos':200,'batch_size':1,
                              'data_aug':True,
                              'transform_methods':[
                                EnsureChannelFirst(),
                                # Data augmentation
                                RandZoom(prob = 0.5, min_zoom=1.0, max_zoom=1.2),
                                RandRotate(range_z = 0.35, prob = 0.8),
                                RandFlip(prob = 0.5),
                                # To tensor
                                ToTensor()
                                ]
                             }
Tr_Pars.loss_par = {'loss':nn.CrossEntropyLoss()}
Tr_Pars.cross_val_par = {'fold':5}
Tr_Pars.model_info = {'model_name':'resnet10','model_type':'classification','experiment_name':'Phase_Detector_test','balanced_sampling':True}
max_epcho = Tr_Pars.model_par['max_epchos']
loss_function = Tr_Pars.loss_par['loss']
tr_transform_methods = Tr_Pars.model_par['transform_methods']


optimizer = torch.optim.Adam(model.parameters(), lr=Tr_Pars.model_par['lr'])


val_transform_methods = [EnsureChannelFirst(), 
                         ToTensor()
                                ]
#set the cross validation
stratify_kfold = StratifiedKFold(n_splits=Tr_Pars.cross_val_par['fold'],shuffle=True,random_state=42)
#set the result path
#set save results


val_interval = 1
plot_interval = 1


for fold,(train_idx,val_idx) in enumerate(stratify_kfold.split(image_files,labels)):
    #define loss list and x_axis list
    tr_result_saver = Utility.SaveResults(SAVE_PATH + str(fold) +'/')
    val_result_saver = Utility.SaveResults(SAVE_PATH + str(fold) + '/',type='val')
    epoch_loss_values, train_loss_epoch_x_axis = [], []
    val_loss_values, val_loss_epoch_x_axis = [], []
    train_images,train_labels = [image_files[i] for i in train_idx],[labels[i] for i in train_idx]
    val_images,val_labels = [image_files[i] for i in val_idx],[labels[i] for i in val_idx ]
    # creat train dataset and dataloader
    train_ds, val_ds = Utility.Image_Dataset(train_images,train_labels,transform_methods=tr_transform_methods),Utility.Image_Dataset(val_images,val_labels,transform_methods=[EnsureChannelFirst(),
                            ToTensor() 
                                ])
    
    # set saver
    #set sampler
    sampler = Utility.Balanced_sampler(labels=train_labels)
    train_loader,val_loader = DataLoader(dataset=train_ds,batch_size=Tr_Pars.model_par['batch_size'],num_workers=0,pin_memory=True,sampler=sampler),DataLoader(dataset=val_ds,batch_size=1,num_workers=0,pin_memory=True)
    for epoch in range(max_epcho):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epcho}")
        print("-" * 10)

        model.train()

        #init loss
        epoch_loss = 0
        val_loss = 0
        step = 0

        # init output and lstart = time.time()abel list for metrics
        train_label = []
        train_pred = []
        train_pred_raw = []


        for im,label in train_loader:
            #contrast
            im = torch.tensor(Utility.convert_hu_to_grayscale(im))
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            #move the data to device
            im_input,label= im.to(device),label.to(device)

            #forward
            output = (model(im_input))
            train_pred_raw.append(output)
            #multiple classification! use softmax

            loss = loss_function(output,label)

            train_label.append(label.cpu())
            train_pred.append(output.cpu())

            loss.backward()#calculate the gradient
            optimizer.step() #update the parameters

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
        
        #log the  loss
        epoch_loss /= step #average loss over the epoch per step
        epoch_loss_values.append(epoch_loss)
        train_loss_epoch_x_axis.append(epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")



    
        #calculate the metrics
        if (epoch + 1) % val_interval == 0:
            metrics = Utility.Metrics(3,train_pred,train_labels)
            AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()
            #store results
            tr_result_saver.store_results(tr_result_saver.df_results(four_rate_dic,AUC,accuracy,F1,epoch_loss,epoch))

        # validate the model
        val_step = 0
        val_label = []
        val_pred = []
        val_pred_raw = []


        for im,label in val_loader:
            model.eval()
            #contrast
            im = torch.tensor(Utility.convert_hu_to_grayscale(im))
            im_input,label= im.to(device),label.to(device)
            
            val_step += 1
            with torch.no_grad():
                output = (model(im_input))
                val_pred_raw.append(output)
                #output = sigmiod(output)
                loss = loss_function(output, label.long())
                # keep tack of the labels and predictions
                val_label.append(label.cpu())
                val_pred.append(output.cpu())
                val_loss += loss.item()
                val_epoch_len = len(val_ds) // val_loader.batch_size # calculate the number of steps in an epoch
        #get average loss
        val_loss /= val_step
        val_loss_values.append(val_loss)
        val_loss_epoch_x_axis.append(epoch + 1)
        metrics = Utility.Metrics(3,val_pred,val_label)
        AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()
        #store results
        val_result_saver.store_results(val_result_saver.df_results(four_rate_dic,AUC,accuracy,F1,val_loss,epoch))
        print(f"epoch {epoch + 1} average loss: {val_loss:.4f}")

        # plot the loss
        if (epoch + 1) % plot_interval == 0:
            Utility.plot(train_loss_epoch_x_axis, epoch_loss_values, val_loss_epoch_x_axis, val_loss_values, tr_result_saver.result_path, epoch + 1)

#save the model
model_info = Tr_Pars.write_model_info(SAVE_PATH+ 'model_info.txt')

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Training script for medical image processing.')

    # Model parameters
    parser.add_argument('-- pars', type=json, default=0.01, help='Learning rate for the optimizer')

    # Paths
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the labels file')
    parser.add_argument('--result_path', type=str, required=True, help='Path to save the results')

    # Data augmentation
    # Other options
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross-validation')
    
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    

