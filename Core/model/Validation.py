import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")


from tqdm import tqdm
from Utils.Utility import visual_input, apply_window_to_volume

import torch
import matplotlib.pyplot as plt
from Utils.Metrics import Metrics

CLASSIFICATION = {'blanco':0,'AP':1,"PVP":2}


def Validation_loop(cfg,model,dataloader,criterion):
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
    #model = model.to(device)
    #predict


    for i,(im,label) in enumerate(vali_bar):       
        #rotate and flip
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)

        if cfg.visual_im.visual_im:
            # visualize input
            visual_input(im,label,cfg.visual_im.visual_out_path)


        im,label = im.to(cfg.SYSTEM.DEVICE),label.to(cfg.SYSTEM.DEVICE)
        #print('validation',im.shape,label)

        with torch.no_grad():
            output = (model(im))

            loss = criterion(output,label)
            average_loss += loss.item()

            #softmax probability
            output = torch.nn.functional.softmax(output,dim=1)
            #print('this is output',output)

    

        #softmax probability
        y_pred.append(output.cpu())
        y_true.extend(label.cpu().numpy().tolist())


        #print("this is y_pred",output,'and this is y_true',label)
        #print("this is step loss",loss)
        
        #set description for tqdm


        vali_bar.set_description(f"label{label},loss:{average_loss},out_put_prob:{output}")

        #metrics
    

    
    average_loss = average_loss/len(vali_bar)
    print('this is average loss',average_loss)
    return average_loss,y_pred,y_true

    