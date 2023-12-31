import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from tqdm import tqdm
import torch

from torch.utils.data import Subset

from Utils.Models import build_model
from Utils.Utility import path_check,visual_input
from Utils.Metrics import Metrics

from Utils.Utility import apply_window_to_volume


def train_loop(cfg,model,dataloader,epoch_num,optimizer,criterion,scheduler=None):
    """
    args:
        model: model to be trained
        dataloader: dataloader
        epoch_num: number of epochs
        device: device to train on
        optimizer: optimizer
        criterion: loss function
        learning_rate: learning rate
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
    for i,(im,label) in enumerate(train_bar):
 

        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)
        #print('this is im',im.shape)
        if cfg.visual_im.visual_im:
            # visualize input
            visual_input(im,label+i,cfg.visual_im.visual_out_path)




        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE)
        #print leraing rate
        #print(f"learning rate: {scheduler.get_last_lr()[0]}")

        optimizer.zero_grad()

        output = (model(im))
        #print('mother fucker loss function',criterion)
        #print(type(label),'and fucking label',label)
        loss = criterion(output,label)
        loss.backward()

        average_loss += loss.item()
        

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output.cpu())
        y_true.extend(label.cpu().numpy().tolist())


        #print(y_true,6666)
        #set description for tqdm
        train_bar.set_description(f"step_loss:{loss}")
        #print(f"y_true_label{label};y_predict:{output};step_loss{loss}")
        optimizer.step()

        #scheduler.step()

        #metrics
    

    #print('accur',accuracy)
    

    average_loss /= len(train_bar)
    #print('average_loss',average_loss)
    return average_loss,y_pred,y_true

    