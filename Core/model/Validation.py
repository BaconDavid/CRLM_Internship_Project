import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")


from tqdm import tqdm
from Utils.Utility import visual_input, apply_window_to_volume

import torch
import matplotlib.pyplot as plt
from Utils.Metrics import Metrics

CLASSIFICATION = {'blanco':0,'AP':1,"PVP":2}


def Validation_loop(model,dataloader,device,num_class,criterion,visual_im,visual_out_path=None):
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
    print("##################")
    print("##################")

    for i,(im,label) in enumerate(vali_bar):
         #applying windowing
        im = apply_window_to_volume(im,50,400)
        im = torch.tensor(im)


        if visual_im:
            # visualize input
            visual_input(im,label,visual_out_path)

        im,label = im.to(device),label.to(device)
        model = model.to(device)
        #predict
        model.eval()
        with torch.no_grad():
            output = (model(im))
            loss = criterion(output,label)
            average_loss += loss.item()

            #softmax probability
            output = torch.nn.functional.softmax(output,dim=1)

    

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output.cpu())
        y_true.append(label.cpu())
        print("this is y_pred",output,'and this is y_true',label)
        print("this is step loss",loss)
        
        #set description for tqdm


        vali_bar.set_description(f"loss:{average_loss}")

        #metrics
    

    
    average_loss = average_loss/len(vali_bar)
    print('this is average loss',average_loss)
    return average_loss,y_pred

    