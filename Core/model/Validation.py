import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")


from tqdm import tqdm
from Utils.Utility import visual_input

import torch
import matplotlib.pyplot as plt

CLASSIFICATION = {'blanco':0,'AP':1,"PVP":2}


def Validation_loop(model,dataloader,device,criterion,visual_input,visual_out_path=None):
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
    loss = 0

    #set metrics record
    y_pred = []
    print("##################")
    print("##################")

    for i,(im,label) in enumerate(vali_bar):
        if visual_input:
            # visualize input
            visual_input(im,label,visual_out_path)

        im,label = im.to(device),label.to(device)
        #predict
        with torch.no_grad():
            output = model(im)
            loss = criterion(output,label)
            loss += loss.item()

            #softmax probability
            output = torch.nn.functional.softmax(output,dim=1)
            y_pred.append(output)
        output = model(im)
        loss = criterion(output,label)
        loss += loss.item()

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output)
        
        #set description for tqdm

        average_loss = loss/(i+1)

        vali_bar.set_description(f"loss:{average_loss}")

    return loss,y_pred

    