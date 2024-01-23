import torch
import pandas as pd
import numpy as np
from monai.metrics import get_confusion_matrix,compute_roc_auc
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from torch import tensor
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,roc_curve,auc,confusion_matrix
class Metrics():
    def __init__(self,num_class=2,y_pred=None,y_true_label=None,targets=None):
        """
        args:
            y_pred: list of predicted tensor
            y_true_label: list of true labels
            targets: dicts of targets and their labels
        """
        self.num_class = num_class
        self.y_pred = np.stack([y.detach().cpu().numpy() for y in y_pred],axis=0).reshape(-1)
        self.four_rate_dic = {str(i):{'tp':0,'fp':0,'tn':0,'fn':0} for i in range(num_class)}
        self.y_true_label = np.array(y_true_label)
        #self.y_pred_label = [torch.argmax(y_pre,dim=1).detach().cpu().numpy().tolist() for y_pre in y_pred]
        #self.y_pred_label = [item for sublist in self.y_pred_label for item in sublist]

        self.y_pred_label = np.round(self.y_pred).reshape(-1).astype(int)
        #self.y_pred_one_hot = torch.nn.functional.one_hot(torch.tensor(self.y_pred_label,dtype=torch.int64),num_classes=self.num_class)
        #self.y_true_one_hot = torch.nn.functional.one_hot(torch.tensor(self.y_true_label.tolist(),dtype=torch.int64),num_classes=self.num_class)
        self.metrics_df = pd.DataFrame()

    def calculate_metrics(self):
        self.metrics = {str(i): {'f1': 0, 'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0} for i in range(self.num_class)}

        # 计算每个类别的指标
        for i in range(self.num_class):
            true_binary = (self.y_true_label == i).astype(int)
            pred_binary = (self.y_pred_label == i).astype(int)

            self.metrics[str(i)]['f1'] = f1_score(true_binary, pred_binary)
            self.metrics[str(i)]['precision'] = precision_score(true_binary, pred_binary)
            self.metrics[str(i)]['recall'] = recall_score(true_binary, pred_binary)

            # 对于二分类问题，使用sigmoid输出的概率来计算AUC
            if len(np.unique(true_binary)) > 1:
                self.metrics[str(i)]['auc'] = roc_auc_score(true_binary, self.y_pred)

            self.metrics[str(i)]['accuracy'] = accuracy_score(true_binary, pred_binary)

        return self.metrics




    def get_roc(self,average='binary'):
        #return compute_roc_auc(self.y_pred_one_hot,self.y_true_one_hot,average)
        #self.y_pred_ = np.stack(self.y_pred,axis=0)[:,:,1].reshape(-1,)
        return roc_auc_score(self.y_true_label,self.y_pred)

    def get_four_rate(self):
    # 计算混淆矩阵
        confu_matrix = confusion_matrix(self.y_true_label, self.y_pred_label)

        # 请确保confu_matrix是2x2矩阵
        if confu_matrix.shape != (2, 2):
            raise ValueError("Confusion matrix must be 2x2 for binary classification.")

        # 提取混淆矩阵的元素
        tn, fp, fn, tp = confu_matrix.ravel()

        # 更新四率字典
        self.four_rate_dic['0']['tp'] = tn
        self.four_rate_dic['0']['fp'] = fp
        self.four_rate_dic['0']['tn'] = tp
        self.four_rate_dic['0']['fn'] = fn

        self.four_rate_dic['1']['tp'] = tp
        self.four_rate_dic['1']['fp'] = fp
        self.four_rate_dic['1']['tn'] = tn
        self.four_rate_dic['1']['fn'] = fn

        return self.four_rate_dic

    def get_accuracy(self) -> float:
        """
        args:
            y_pred_label: list of predicted labels
            y_true_label: list of true labels
        """
        accuracy = accuracy_score(self.y_pred_label,self.y_true_label)
        return accuracy
    
    def get_f1_score(self,average='macro') -> float:
        return f1_score(self.y_true_label,self.y_pred_label,average=average)
    

    def generate_metrics_df(self, epoch):
        # 存储度量数据
        metrics_data = []
        for class_id, class_metrics in self.metrics.items():
            data_row = {"epoch": epoch}  # 首先添加 epoch
            data_row.update({"class_id": class_id})  # 然后添加 class_id
            data_row.update(class_metrics)  # 最后添加其他指标
            metrics_data.append(data_row)

        # 将新数据添加到现有的DataFrame中
        new_df = pd.DataFrame(metrics_data)
    # Using concat instead of append
        #self.metrics_df = pd.concat([self.metrics_df, new_df], ignore_index=True)

        return new_df