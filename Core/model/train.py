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


def train_loop(model,dataloader,epoch_num,device,num_class,optimizer,scheduler,criterion,visual_im=True,leraning_rate=0.01,visual_out_path=None,debug=False):
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
        #print('mother fucker',im.shape)
        
        #applying windowing
       
        if visual_im:
            # visualize input
            visual_input(im,label+i,visual_out_path)





        im,label = im.to(device),label.to(device)
        #print leraing rate
        print(f"learning rate: {scheduler.get_last_lr()[0]}")


        optimizer.zero_grad()

        output = (model(im))
        print('mother fucker loss function',criterion)
        print(type(label),'and fucking label',label)
        loss = criterion(output,label)
        loss.backward()

        average_loss += loss.item()
        

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output.cpu())
        y_true.append(label.cpu())
        #set description for tqdm
        train_bar.set_description(f"step_loss:{loss},learning_rate: {scheduler.get_last_lr()[0]}")
        print(f"y_true_label{label};y_predict:{output};step_loss{loss}")
        optimizer.step()
        scheduler.step()

        #metrics
    

    #print('accur',accuracy)
    

    average_loss /= len(train_bar)
    print('average_loss',average_loss)
    return average_loss,y_pred

    