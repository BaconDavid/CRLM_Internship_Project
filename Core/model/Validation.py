from re import L
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")


from tqdm import tqdm
from Utils.Utility import visual_input, apply_window_to_volume

import torch
import matplotlib.pyplot as plt
from Utils.Metrics import Metrics

CLASSIFICATION = {'blanco':0,'AP':1,"PVP":2}


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
                loss = criterion(output,label)
                average_loss += loss.item()
                output = torch.nn.functional.softmax(output,dim=1)

            elif cfg.MODEL.task == 'selective':
                if cfg.LOSS.SelectiveLoss.loss == 'GamblerLoss':
                    gambler_vali = GamblerValidation(model,epoch_num,criterion,cfg)
                    average_loss,output = gambler_vali.validate(im,label,i,sample_num)                    
                    
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
                        loss, loss_dict = criterion(out_class_accum, out_select_accum,out_aux_accum,label_accum)
                        average_loss += loss.item()
                        accumulated_outputs.clear(),accumulated_labels.clear() #clear accumulation list

                    #loss,loss_dict = criterion(out_class,out_select,out_aux,label)
                    out_class = torch.nn.functional.softmax(out_class,dim=1)
                    out_aux = torch.nn.functional.softmax(out_aux,dim=1)
                    output = (out_class,out_select,out_aux)
                    print(i,'epoch')
            #print('this is output',output)
            else:
                pass
    


        y_pred.append(output)
        y_true.extend(label.cpu().numpy().tolist())


        #set description for tqdm


        vali_bar.set_description(f"label{label},loss:{average_loss},out_put_prob:{output}")

    average_loss = (average_loss * accumulated_batch)/ len(vali_bar)

    print('this is average loss',average_loss)
    print('return',y_pred)
    return average_loss,y_pred,y_true



class GamblerValidation:
    def __init__(self, model, epoch_num, criterion, cfg):
        self.model = model
        self.epoch_num = epoch_num
        self.criterion = criterion
        self.cfg = cfg

    def validate(self, im, label,sample_index,sample_num):
        average_loss = 0
        self.model.eval()  # 确保模型处于训练模式
        

        if self.cfg.MODEL.pretrained and (self.epoch_num < self.cfg.MODEL.Gambler.pretrain_epochs):
            print('Pretrain loop!')
            output = self.model(im)
            # 仅提取0,1类别进行交叉熵损失计算
            loss = torch.nn.CrossEntropyLoss()(output[:, :-1], label)

        else:
            output = self.model(im)
            loss = self.criterion(output, label)


        if ((sample_index + 1) % self.cfg.TRAIN.batch_accumulation_size == 0) or (sample_index == sample_num - 1):
            loss /= self.cfg.TRAIN.batch_accumulation_size # average loss
            average_loss += loss.item()

        output = torch.nn.functional.softmax(output, dim=1)



        return   average_loss,output