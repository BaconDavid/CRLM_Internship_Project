import sys
sys.path.append('../')
import os
import SimpleITK as sitk
import numpy as np

from Core.Dataset.Dataloader import Data_Loader,Image_Dataset,DataFiles
from Core.Utils.Models import build_model

# Import M3d-CAM
from medcam import medcam
import torch
from monai.networks.nets import resnet10
from monai.transforms import ToTensor,EnsureChannelFirst,Compose,Resize

from Core.Config.config import get_cfg_defaults
from scipy.ndimage import zoom as zoom

import argparse

def load_model(cfg):
    model = build_model(cfg)
    return model



class LoadData:
    def __init__(self,cfg,mode='test') -> None:
        self.cfg = cfg
        if mode == 'test':
            self.path = cfg.TEST.test_path
            self.path = cfg.TEST.test_label_path
        else:
            self.path = cfg.VALID.valid_path
            self.path = cfg.VALID.valid_label_path
            
        self.label_name = cfg.LABEL.label_name

    def load_data(self):
        label_name = self.cfg.LABEL.label_name
        data = DataFiles(self.path,self.label_path,label_name)
        images_lst = sorted(data.get_image_files())
        labels_lst = data.get_label_files()
        data.Data_check()

        return images_lst,labels_lst
    
    def get_data_loader(self,transform_methods=None):
        images_lst,labels_lst = self.load_data()
        dataset = Image_Dataset(images_lst,labels_lst,transform_methods)
        data_loader = Data_Loader(dataset=dataset,num_workers=self.cfg.SYSTEM.NUM_WORKERS,batch_size=self.cfg.TRAIN.batch_size).build_train_loader() 
        
        return data_loader



def get_attention_map(cfg,label=0,mode='valid',weight_path=None,backend='gcam',layer='layer4',save_maps=True,return_attention=True):
    Mymodel = load_model(cfg)
    data_loader = LoadData(cfg,mode).get_data_loader()
    
    #load weight
    Weight_dict = torch.load(weight_path)
    Mymodel.load_state_dict(Weight_dict['model'])

    Mymodel = medcam.inject(Mymodel, output_dir=f"./attention_maps/{cfg.MODEL.name}", backend='gcam', layer='layer4', label=0, save_maps=True,return_attention=True)
    Mymodel.eval()

    for i,(im,label) in enumerate(data_loader):
        im = torch.rot90(im,k=3,dims=(2,3))
        im = torch.flip(im,[3])
        #permute to [B,C,D,H,W]
        im = im.permute(0,1,4,2,3)
        Mymodel(im)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',type=str,required=True,help='path to config file')
    parse_args.add_argument('--weight_path',type=str,required=True,help='path to weight file')
    parse_args.add_argument('--mode',type=str,default='valid',help='train/vali or test')
    parse_args.add_argument('--label',type=int,default=0,help='label')
    parse_args.add_argument('--backend',type=str,default='gcam',help='gcam or gradcam')
    args = parser.parse_args()
    return args


def main():
    args = args.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg.TRAIN.lr,type(cfg.TRAIN.lr))
    print('successfully load the config file !')
    get_attention_map(cfg,label=args.label,mode=args.mode,weight_path=args.weight_path,backend=args.backend)
    

        