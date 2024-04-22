from collections import OrderedDict
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
    print(len(train_bar),'length of train_bar')
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
    #model = model.to(device)
    for i,data in enumerate(train_bar):
        if cfg.DATASET.mask:
            im,label,_,mask = data
            #stack channel
            im = torch.cat((im,mask),dim=1)
            #print('im',im.shape)
            
        else:
            im,label,_ = data

        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)

        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE) # to device

        
        label = label.long()

        ##TRAIN by task    
        optimizer.zero_grad()
        if cfg.MODEL.task == 'selective':

            if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                gambler_train = GamblerTrain(model,epoch_num,optimizer,criterion,scheduler,cfg)
                average_loss, output = gambler_train.train(im,label,i,sample_length)
                

            elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                
                output = model(im)
                
                out_class,out_select,out_aux = output
                accumulated_outputs.append(output), accumulated_labels.append(label)
                # loss dict includes, 'empirical_risk' / 'emprical_coverage' / 'penulty'

                
                
                #softmax for class and aux
                out_class = torch.nn.functional.softmax(out_class,dim=1)
                out_aux = torch.nn.functional.softmax(out_aux,dim=1)

                output = (out_class,out_select,out_aux)
                
                # every 5 batch gradient accumulation
                if (i+1) % accumulated_batch ==0:
                    out_class_accum = torch.cat([out[0] for out in accumulated_outputs], dim=0)
                    out_select_accum = torch.cat([out[1] for out in accumulated_outputs], dim=0)
                    out_aux_accum = torch.cat([out[2] for out in accumulated_outputs], dim=0)
                    label_accum = torch.cat(accumulated_labels, dim=0)
                    print('accumulated',out_class_accum,out_select_accum,out_aux_accum)
                    loss, loss_dict = criterion(out_class_accum, out_select_accum,out_aux_accum,label_accum)
                    loss.backward()
                    optimizer.step()
                    average_loss += loss.item()
                    #clear accumulation list
                    accumulated_outputs.clear(),accumulated_labels.clear()
            else:
                raise ValueError('SelectiveLoss can only be GamblerLoss or SelectiveLoss')
            
        elif cfg.MODEL.task == 'classification':
            output = model(im)
            #print('output',output.shape,label.shape)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            output = torch.nn.functional.softmax(output,dim=1)
            

        #softmax probability
        y_pred.append(output)
        y_true.extend(label.cpu().numpy().tolist())
        #set description for tqdm
        train_bar.set_description(f"label:{label},lr:{optimizer.param_groups[0]['lr']},out_put_prob:{output}")
        #print(f"y_true_label{label};y_predict:{output};step_loss{loss}")

        if scheduler:
            print('scheduler stepup')
            scheduler.step()
        ema.update()


        #scheduler.step()

        #metrics
    

    #print('accur',accuracy)
    

    average_loss = (average_loss * accumulated_batch)/ len(train_bar)
    print('average_loss',average_loss)
    return average_loss,y_true,y_pred

class GamblerTrain:
    def __init__(self, model, epoch_num, optimizer, criterion, scheduler, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cfg = cfg

    def train(self, im, label,sample_index,sample_num):
        average_loss = 0

        self.model.train()  # 确保模型处于训练模式
        

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

        if ((sample_index + 1) % self.cfg.TRAIN.batch_accumulation_size == 0) or (sample_index == sample_num - 1):
            loss /= self.cfg.TRAIN.batch_accumulation_size # average loss
            average_loss += loss.item()
            #update
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        output = torch.nn.functional.softmax(output, dim=1)



        return   average_loss,output