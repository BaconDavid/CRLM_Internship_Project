import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

import tqdm
from Utils.Models import build_model
from Utils.Utility import path_check,visual_input
import loss,optimizer,scheduler

def train_loop(model,dataloader,epoch_num,device,optimizer,criterion,visual_input=False,leraning_rate=0.01,visual_out_path=None):
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

    #set loss,optmizer and scheduler
    loss = loss.build_loss()
    optimizer = optimizer.build_optimizer(model)

    print("##################")
    print(f"epoch {epoch_num}")
    print("##################")

    for i,(im,label) in enumerate(train_bar):
        if visual_input:
            # visualize input
            visual_input(im,label,visual_out_path)

        label = label.to(device)
        optimizer.zero_grad()
        output = model(im)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        train_bar.set_description(f"loss: {loss.item():.5f}")

    