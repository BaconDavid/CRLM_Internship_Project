from  Core.model import build_model
from Core.model import loss,optimizer,scheduler
from Core.model.train import train_loop,Validation,test
from Core.Dataset.Dataloader import Image_Dataset,Data_Loader
from Core.Utils.Utility import DataFiles,path_check

import os

def main(data_path,epochs,mode='train'):
    """
    args:
        epochs: number of epochs
        mode: train/vali or test
        data_path: path to the data folder

    """


    Data = DataFiles(data_path,)

    if mode == 'train':

        model = build_model('resnet10',3)


        #set scheduler,optimizer parameters

        optimizer_param = {"lr":0.001}
        scheduler_param = {"step_size":10,"gamma":0.1}

        loss_fun = loss.build_loss()
        scheduler_fun = scheduler.build_scheduler(scheduler_param) 
        optimizer_param = {"lr":0.001}
        optimizer_fun = optimizer.build_optimizer(model.parameters(),optimizer_param)


        #prepare data for training
        dataloader = Image_Dataset()

        for epoch in range(epochs):
            train_loop(model,dataloader,epoch,device,optimizer_fun,scheduler_fun,loss_fun,visual_input=True,visual_out_path=visual_out_path)
            #save model
            path_check(model_save_path)
            torch.save(model.state_dict(),model_save_path)
            print("model saved")
    elif mode == 'test':
        test_loop(model,dataloader,device,loss_fun,visual_input=True,visual_out_path=visual_out_path)


