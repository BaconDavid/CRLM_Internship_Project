from re import L
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")


from tqdm import tqdm
from Utils.Utility import visual_input, apply_window_to_volume

import torch
import matplotlib.pyplot as plt
from Utils.Metrics import Metrics

def Validation_loop(cfg,model,dataloader,criterion,epoch_num):
    """
    args:
        model: model to be trained
        dataloader: dataloader
        device: device to train on
        criterion: loss function
        visual_input: visualize input
    """
    #prepare data for training
    vali_bar = tqdm(dataloader)
    average_loss = 0

    #set metrics record
    y_pred = []
    y_true = []
    
    #only for selectivenet to store batch samples
    accumulated_outputs = []
    accumulated_labels = []
    accumulated_batch = cfg.VALID.batch_accumulation_size
    sample_num = len(vali_bar)
    print("##################")
    print("##################")
    #model = model.to(device)
    #predict

    validator = build_validator(model, epoch_num, criterion, cfg)
    for i,data in enumerate(vali_bar):
        if cfg.DATASET.mask:
            im,label,_,mask = data
            #add mask to channel
            im = torch.cat((im,mask),dim=1)
        else:
            im,label,_ = data    

        #   
        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)

        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE)

        with torch.no_grad():
            if cfg.MODEL.task == 'classification':
                output = (model(im))
                #loss = criterion(output,label)
                average_loss_valid,output = validator.grad_accumulate(im,label)
                #print('average_loss in validation one epoch!',average_loss_valid)
                #print('output in validation one epoch!',output)
                average_loss += average_loss_valid
                average_loss_dict = None

            elif cfg.MODEL.task == 'selective':
                if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                    average_loss_valid,output = validator.grad_accumulate(im,label)
                    average_loss += average_loss_valid
                    average_loss_dict = None      

                elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
                    output = model(im)
                    out_class,out_select,out_aux = output
                    accumulated_outputs.append(output),accumulated_labels.append(label)
                    #calculate batch accumulation
                    if (i+1) % accumulated_batch ==0:
                        out_class_accum = torch.cat([out[0] for out in accumulated_outputs], dim=0)
                        out_select_accum = torch.cat([out[1] for out in accumulated_outputs], dim=0)
                        out_aux_accum = torch.cat([out[2] for out in accumulated_outputs], dim=0)
                        label_accum = torch.cat(accumulated_labels, dim=0)
                        #print('accumulated',out_class_accum,out_select_accum,out_aux_accum)
                        average_loss_valid, average_loss_dict = validator.grad_accumulate(out_class_accum, out_select_accum,out_aux_accum,label_accum)
                        average_loss += average_loss_valid
                        accumulated_outputs.clear(),accumulated_labels.clear() #clear accumulation list

                    #loss,loss_dict = criterion(out_class,out_select,out_aux,label)
                    out_class = torch.nn.functional.softmax(out_class,dim=1)
                    out_aux = torch.nn.functional.softmax(out_aux,dim=1)
                    output = (out_class,out_select,out_aux)

        y_pred.append(output)
        y_true.extend(label.cpu().numpy().tolist())
        vali_bar.set_description(f"label{label},loss:{average_loss}")

    average_loss = average_loss / len(vali_bar)

    print('this is average loss',average_loss)
    return average_loss,y_pred,y_true,average_loss_dict

def build_validator(model, epoch_num, criterion, cfg):
    if cfg.MODEL.task == 'selective':
        if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
            return GamblerValidate(model,epoch_num,criterion,cfg)
        elif cfg.LOSS.SelectiveLoss.loss == 'SelectiveLoss':
            return SelectiveValidate(model, epoch_num, criterion, cfg)
    elif cfg.MODEL.task == 'classification':
        return ClassificationValidate(model, epoch_num, criterion, cfg)

class GamblerValidate:
    def __init__(self, model, epoch_num, criterion, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.criterion = criterion
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
            loss = torch.nn.CrossEntropyLoss()(output[:, :-1], label)
        else:
            output = self.model(im)
            loss,loss_dict = self.criterion(output, label)
            
        loss_value = loss.item()
        output = torch.nn.functional.softmax(output,dim=1)
        return loss_value, output


    
class SelectiveValidate:
    def __init__(self, model, epoch_num, criterion,  cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.criterion = criterion
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
        average_loss = loss.item() * self.cfg.TRAIN.batch_accumulation_size #every batch size calculate loss so multiple batch size
        return average_loss,loss_dict
    

    

class ClassificationValidate:
    def __init__(self, model, epoch_num, criterion,cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.criterion = criterion
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
        loss = self.criterion(output, label)
        loss_value = loss.item()
        output = torch.nn.functional.softmax(output,dim=1)
        return loss_value, output





