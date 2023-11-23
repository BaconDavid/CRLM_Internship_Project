import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

import tqdm
from Utils.Models import build_model
from Utils.Utility import path_check,visual_input
from Utils.Metrics import Metrics


NUM_CLASSES = 2

def train_loop(model,dataloader,epoch_num,device,optimizer,scheduler,criterion,visual_input=False,leraning_rate=0.01,visual_out_path=None):
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
    loss = 0

    #set metrics record
    y_pred = []
    print("##################")
    print(f"epoch {epoch_num}")
    print("##################")

    for i,(im,label) in enumerate(train_bar):
        if visual_input:
            # visualize input
            visual_input(im,label,visual_out_path)

        im,label = im.to(device),label.to(device)
        #print leraing rate
        print(f"learning rate: {scheduler.get_last_lr()[0]}")


        optimizer.zero_grad()
        output = model(im)
        loss = criterion(output,label)
        loss.backward()
        loss += loss.item()

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output)
        
        #set description for tqdm

        average_loss = loss/(i+1)

        train_bar.set_description(f"loss:{average_loss},learning_rate: {scheduler.get_last_lr()[0]}")
        optimizer.step()
        scheduler.step()
    

    return loss,y_pred

    