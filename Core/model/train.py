import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")

from tqdm import tqdm
import torch


from Utils.Models import build_model
from Utils.Utility import path_check,visual_input
from Utils.Metrics import Metrics


NUM_CLASSES = 2

def train_loop(model,dataloader,epoch_num,device,num_class,optimizer,scheduler,criterion,visual_im=False,leraning_rate=0.01,visual_out_path=None):
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

    #set metrics record
    y_pred = []
    y_true = []
    print("##################")
    print(f"epoch {epoch_num}")
    print("##################")

    for i,(im,label) in enumerate(train_bar):
        #print('mother fucker',im.shape)
        if visual_im:
            # visualize input
            visual_input(im,label,visual_out_path)

        im,label = im.to(device),label.to(device)
        #print leraing rate
        print(f"learning rate: {scheduler.get_last_lr()[0]}")


        optimizer.zero_grad()
        model.train()
        output = (model(im))
        print('mother fucker loss function',criterion)
        print(type(label),'and fucking label',label)
        loss = criterion(output,label)
        loss.backward()
        average_loss += loss

        #softmax probability
        output = torch.nn.functional.softmax(output,dim=1)
        y_pred.append(output.cpu())
        y_true.append(label.cpu())
        #set description for tqdm
        print('this is y_true label list',y_true)
        print('this is fucking y_pred',output,";and y_true",label)
        train_bar.set_description(f"step_loss:{loss},learning_rate: {scheduler.get_last_lr()[0]}")
        optimizer.step()
        scheduler.step()

        #metrics
        metrics = Metrics(num_class,y_pred,y_true)
        print('this is fucking metric',metrics.y_true_label,metrics.y_pred_label)
        AUC,accuracy,F1,four_rate_dic = metrics.get_roc(),metrics.get_accuracy(),metrics.get_f1_score(),metrics.get_four_rate()
        print('accur',accuracy)
    

    average_loss /= len(train_bar)
    print('fucking average_loss',average_loss)
    return average_loss,y_pred,AUC,accuracy,F1,four_rate_dic

    