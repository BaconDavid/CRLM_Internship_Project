import sys
import os
from numpy import mask_indices
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from tqdm import tqdm
import torch

from torch.utils.data import Subset

from Utils.Utility import path_check,visual_input
from Utils.Metrics import Metrics

from Utils.Utility import apply_window_to_volume
from ema_pytorch import EMA



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
    average_loss = 0
    print(len(train_bar),'length of train_bar')
    #set metrics record
    y_pred = []
    y_true = []
    print("##################")
    print(f"epoch {epoch_num+1}")
    print("##################")
    #model = model.to(device)
    for i,data in enumerate(train_bar):
        if cfg.DATASET.mask:
            im,label,_,mask = data
            #stack channel
            im = torch.cat((im,mask),dim=1)
        else:
            im,label,_ = data
    
        
        #make the same as the model output
        if cfg.MODEL.task == 'regression':
            label = label.unsqueeze(1)

        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)



        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE)
        im,label = im.float(),label.float()
        optimizer.zero_grad()
        

        output = (model(im))
        print('output',output.shape,label.shape)
        loss = criterion(output,label)
        loss.backward()


        average_loss += loss.item()
        if cfg.MODEL.task == 'classification':
            output = torch.nn.functional.softmax(output,dim=1)
        else:
            output = torch.nn.functional.sigmoid(output)
        #softmax probability
        y_pred.append(output.cpu())
        y_true.extend(label.cpu().numpy().tolist())
        #set description for tqdm
        train_bar.set_description(f"label:{label},lr:{optimizer.param_groups[0]['lr']},step_loss:{loss},out_put_prob:{output}")
        #print(f"y_true_label{label};y_predict:{output};step_loss{loss}")
        optimizer.step()
        if scheduler:
            print('scheduler stepup')
            scheduler.step()
        ema.update()


        #scheduler.step()

        #metrics
    

    #print('accur',accuracy)
    

    average_loss /= len(train_bar)
    print('average_loss',average_loss)
    return average_loss,y_pred,y_true

