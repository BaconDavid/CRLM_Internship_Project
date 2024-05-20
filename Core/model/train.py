from collections import OrderedDict
from logging import critical
import sched
import sys
import os
from numpy import average, mask_indices
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import Subset

from Utils.Utility import path_check,visual_input
from Utils.Metrics import Metrics

from Utils.Utility import apply_window_to_volume
from ema_pytorch import EMA

import numpy as np


def train_loop(cfg,model,dataloader,epoch_num,optimizer,criterion,ema=None,scheduler=None):
    """
    args:
        cfg: cfg configuration file
        model: model
        dataloader: dataloader
        epoch_num: epoch number
        optimizer: optimizer
        criterion: loss function
        ema: exponential moving average
        scheduler: scheduler
        
    """
    #prepare data for training
    train_bar = tqdm(dataloader)
    sample_length = len(dataloader)
    average_loss = 0

    #set metrics record
    y_pred = []
    y_true = []

    #only for selectivenet to store batch samples
    accumulated_outputs = []
    accumulated_labels = []
    accumulated_batch = cfg.TRAIN.batch_accumulation_size

    print("##################")
    print(f"epoch {epoch_num+1}")
    print("##################")

    optimizer.zero_grad() #clear grad
    trainer = build_trainer(model,epoch_num,optimizer,criterion,scheduler,cfg)

    for i,data in enumerate(train_bar):
        if cfg.DATASET.mask:
            im,label,_,mask = data
            #stack channel
            im = torch.cat((im,mask),dim=1)
        else:
            im,label,_ = data

        #rotate and flip to make shape of [B,C,D,H,W]
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)
        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE) # to device
        label = label.long()

        ##TRAIN by task    
        if cfg.MODEL.task == 'selective':
            if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                average_loss_train, output = trainer.grad_accumulate(im,label) #accmulated loss 
                if ((i + 1) % cfg.TRAIN.batch_accumulation_size == 0):
                    trainer.update()
                average_loss += average_loss_train


            elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                output = model(im)
                out_class,out_select,out_aux = output
                accumulated_outputs.append(output), accumulated_labels.append(label)
               
                #softmax for class and aux
                out_class = torch.nn.functional.softmax(out_class,dim=1)
                out_aux = torch.nn.functional.softmax(out_aux,dim=1)
                output = (out_class,out_select,out_aux)
                
                # every 5 batch gradient accumulation
                if ((i + 1) % cfg.TRAIN.batch_accumulation_size == 0) or (i == sample_length - 1):
                    out_class_accum = torch.cat([out[0] for out in accumulated_outputs], dim=0)
                    out_select_accum = torch.cat([out[1] for out in accumulated_outputs], dim=0)
                    out_aux_accum = torch.cat([out[2] for out in accumulated_outputs], dim=0)
                    label_accum = torch.cat(accumulated_labels, dim=0)
                    average_loss_train = trainer.grad_accumulate(out_class_accum,out_select_accum,out_aux_accum,label_accum)
                    trainer.update()
                    accumulated_outputs.clear(),accumulated_labels.clear() #clear accumulation list
                    average_loss += average_loss_train
                    print(average_loss,'this is average loss')

            else:
                raise ValueError('SelectiveLoss can only be GamblerLoss or SelectiveLoss')
            
        elif cfg.MODEL.task == 'classification':
            average_loss_train,output = trainer.grad_accumulate(im,label)
            average_loss += average_loss_train

            if ((i + 1) % cfg.TRAIN.batch_accumulation_size == 0):
                trainer.update()
                
        #store output
        y_pred.append(output)
        y_true.extend(label.cpu().numpy().tolist())
        #set description for tqdm
        train_bar.set_description(f"label:{label},lr:{optimizer.param_groups[0]['lr']})")
        #print(f"y_true_label{label};y_predict:{output};step_loss{loss}")

        # if scheduler:
        #     print('scheduler stepup')
        #     scheduler.step()
        ema.update()


        #scheduler.step()

        #metrics
    

    #print('accur',accuracy)
    
    print(len(train_bar),accumulated_batch,'before finaly average loss')
    average_loss = average_loss / len(train_bar)
    print('average_loss',average_loss)
    return average_loss,y_true,y_pred



def build_trainer(model,epoch_num,optimizer,criterion,scheduler,cfg):
    if cfg.MODEL.task == 'selective':
        if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
            return GamblerTrain(model,epoch_num,optimizer,criterion,scheduler,cfg)
        elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
            return SelectiveTrain(model,epoch_num,optimizer,criterion,scheduler,cfg)
    elif cfg.MODEL.task == 'classification':
        return ClassificationTrain(model,epoch_num,optimizer,criterion,scheduler,cfg)


class GamblerTrain:
    def __init__(self, model, epoch_num, optimizer, criterion, scheduler, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg

    def grad_accumulate(self, im, label):
        """
        args:
            im: image
            label: true label
            sample_index: batch accumulation index
            sample_num: how many samples in total(length of dataloader)
        """
        if self.cfg.MODEL.pretrained and (self.epoch_num < self.cfg.MODEL.Gambler.pretrain_epochs):
            print('Pretrain loop!')
            output = self.model(im)
            # 仅提取0,1类别进行交叉熵损失计算
            loss = nn.CrossEntropyLoss()(output[:, :-1], label)
            loss.backward()
        else:
            output = self.model(im)
            loss = self.criterion(output, label)
            loss.backward()

        loss_value = loss.item()
        output = torch.nn.functional.softmax(output,dim=1)
        return loss_value, output

    def update(self):
        #update
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()

    
class SelectiveTrain:
    def __init__(self, model, epoch_num, optimizer, criterion, scheduler, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg
    
    def grad_accumulate(self,
              out_class_accum,
              out_select_accum,
              out_aux_accum,
              label_accum):
        """
        args:
            out_class_accum: accmulation of prediction head
            out_select_accum: accmulation of output of selection head
            out_aux_accum: accumulation of output of aux head
            label_accum: accumlation of true labels
        """
        average_loss = 0
        self.model.train()

        loss, loss_dict = self.criterion(out_class_accum,out_select_accum,out_aux_accum,label_accum)
        loss.backward()
        average_loss = loss.item() * self.cfg.TRAIN.batch_accumulation_size #every batch size calculate loss so multiple batch size
        return average_loss
    

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler:
            self.scheduler.step()
        #print(out_class_accum.shape,out_aux_accum.shape,out_select_accum.shape,'shape of each part')
        self.optimizer.zero_grad()

class ClassificationTrain:
    def __init__(self, model, epoch_num, optimizer, criterion, scheduler, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg
    
    def grad_accumulate(self,im,label):
        """
        args:
            im: image
            label: true label
            sample_index: batch accumulation index
            sample_num: how many samples in total(length of dataloader)
        """
        output = self.model(im)
        print('here it is!')
        loss = self.criterion(output, label)
        loss.backward()

        loss_value = loss.item()
        output = torch.nn.functional.softmax(output,dim=1)
        return loss_value, output

    def update(self):
        #update
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()




