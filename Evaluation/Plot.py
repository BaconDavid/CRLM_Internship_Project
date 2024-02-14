import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('./')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from Core.Utils.Swin_Transformer_Classification import Swintransformer
import sys
sys.path.append('../')
from Core.Dataset.Dataloader import *
from torch.utils.data import Subset
from Core.Utils.Utility import Balanced_sampler
import argparse

class Evaluation:
    def __init__(self,tr_file,val_file,val_four_rate,save_path) -> None:
        self.tr_df = pd.read_csv(tr_file)
        self.val_df = pd.read_csv(val_file)
        #self.val_metric_df = pd.read_csv(val_four_rate)
        self.save_path = save_path

    def plot_loss(self):
        tr_loss = self.tr_df['loss'].values
        val_loss = self.val_df['loss'].values
        val_accuracy= self.val_df['accuracy'].values
        plt.plot(tr_loss,label='train')
        plt.plot(val_loss,label='val')
        plt.plot(val_accuracy,label='accuracy')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.save_path +'loss.png')
        plt.legend()
        plt.show()

    def plot_auc(self):
        tr_auc = self.tr_df['roc_auc'].values
        val_auc = self.val_df['roc_auc'].values
        plt.plot(tr_auc,label='train')
        plt.plot(val_auc,label='val')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('AUC Values')
        plt.savefig(self.save_path + 'auc.png')

    def plot_recall(self):
        val_recall = self.val_df['recall'].values
        val_precision = self.val_df['precision'].values
        plt.plot(val_recall,label='recall')
        plt.plot(val_precision,label='precision')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('recall-precision')
        plt.show()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_result',type=str,default='train',required=True,help='train/vali or test')
        parser.add_argument('--valid_result',type=str,required=True,help='path to config file')
        parser.add_argument('--valid_four_rate',type=str,required=False,help='four rate of valid')
        parser.add_argument('--save_path',type=str,required=False,help='save path of plots')
        args = parser.parse_args()
        return args
    def main():
        args = parse_args()
        eval = Evaluation(args.train_result,args.valid_result,args.valid_four_rate,args.save_path)
        eval.plot_loss()
        eval.plot_auc()

    main()