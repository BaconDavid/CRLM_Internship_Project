import torch
import pandas as pd
import numpy as np
from monai.metrics import get_confusion_matrix,compute_roc_auc
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from torch import tensor
from sklearn.metrics import precision_score, recall_score,balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Core.model.loss import Loss
class Metrics():
    def __init__(self,y_true,model_output,ave_loss,num_class=2,):
        """
        args:
            model_output: list of model output 
            y_true_label: list of true labels
            targets: dicts of targets and their labels
        """
        self.num_class = num_class
        self.y_pred = model_output
        self.ave_loss = ave_loss
        self.y_true = y_true
        
        
        

        #turn into (steps,batch,out_class) prob
        #[batch,sample,class]


    def generate_metrics_df(self, epoch):
        # 存储度量数据
        metrics_data = []
        for class_id, class_metrics in self.metrics.items():
            data_row = {"epoch": epoch}
            data_row.update({"class_id": class_id})
            data_row.update(class_metrics)
            metrics_data.append(data_row)
        
        # 将新数据添加到现有的DataFrame中
        new_df = pd.DataFrame(metrics_data)
        # Using concat instead of append
        return new_df

    def generate_four_rate_df(self,epoch):
        four_rate_data = []
        for class_id, class_rate in self.four_rate_dic.items():
            data_row = {"epoch": epoch}
            data_row.update({"class_id": class_id})
            data_row.update(class_rate)
            four_rate_data.append(data_row)
        new_df = pd.DataFrame(four_rate_data)
        return new_df
    




class ClassificationMetrics(Metrics):
    def __init__(self,y_true,model_output,ave_loss,num_class=2):
        super().__init__(y_true,model_output,ave_loss,num_class=num_class)
        self.y_pred, self.y_true = self._get_array()#get pred,y_true array
        self.y_pred_label = np.argmax(self.y_pred,axis=2) #predict label

        #get one hot
        self.y_true_one_hot = np.eye(self.num_class)[self.y_true.reshape(-1)]
        self.y_pred_one_hot = np.eye(self.num_class)[self.y_pred_label.reshape(-1)]
        #get metrics
        self.four_rate_dic = {str(i):{'tp':0,'fp':0,'tn':0,'fn':0} for i in range(num_class)}

    def calculate_metrics(self):
        self.metrics = {
            str(i): {'f1': 0, 'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0,'loss':self.ave_loss,'balanced_accuracy':0} for i in range(self.num_class)
        }

        for i in range(self.num_class):
            true_binary = (self.y_true == i).astype(int)
            pred_binary = (self.y_pred_label == i).astype(int)

            
            self.metrics[str(i)]['f1'] = f1_score(true_binary, pred_binary)
            self.metrics[str(i)]['precision'] = precision_score(true_binary, pred_binary)
            self.metrics[str(i)]['recall'] = recall_score(true_binary, pred_binary)
            self.metrics[str(i)]['accuracy'] = accuracy_score(true_binary, pred_binary)
            self.metrics[str(i)]['balanced_accuracy'] = balanced_accuracy_score(true_binary, pred_binary)

            if len(np.unique(true_binary)) > 1:
                self.metrics[str(i)]['auc'] = roc_auc_score(true_binary, self.y_pred[:,:,i].reshape(-1))

           

        return self.metrics

    def get_four_rate(self):
        y_pred_one_hot_tensor = torch.tensor(self.y_pred_one_hot)
        y_true_one_hot_tensor = torch.tensor(self.y_true_one_hot)
        confu_matrix = get_confusion_matrix(y_pred_one_hot_tensor,y_true_one_hot_tensor)
        for i in range(self.num_class):
            self.four_rate_dic[str(i)]['tp'] += confu_matrix[:,i,0].sum().item()
            self.four_rate_dic[str(i)]['fp'] += confu_matrix[:,i,1].sum().item()
            self.four_rate_dic[str(i)]['tn'] += confu_matrix[:,i,2].sum().item() 
            self.four_rate_dic[str(i)]['fn'] += confu_matrix[:,i,3].sum().item()

        return self.four_rate_dic

    def get_roc(self):
        #always return AUC with dHGP even thought it is not a binary classification
        positive_class = self.num_class - 1
        return roc_auc_score(self.y_true_one_hot[:,positive_class],self.y_pred[:,:,positive_class].reshape(-1))
    
    def get_accuracy(self):
        return accuracy_score(self.y_true,self.y_pred_label)
    
    def get_f1_score(self):
        return f1_score(self.y_true,self.y_pred_label)
    
    def _get_array(self):
        # get y_pred numpy with shape (Sample, Batch, Class)
        self.y_pred = np.stack([y.detach().cpu().numpy() for y in self.y_pred],axis=0)
        self.y_true = np.array(self.y_true)
        return self.y_pred,self.y_true






class SelectiveMetrics(Metrics):
    def __init__(self,y_true,model_output,ave_loss,num_class=2,coverage=None,loss_type='GamblerLoss'):
        super().__init__(y_true,model_output,ave_loss,num_class=num_class)
        #separate another extra class head!
        self.loss_type = loss_type
        assert loss_type in ['GamblerLoss','SelectiveLoss']


        if loss_type == 'GamblerLoss':
            self.y_pred, self.y_true = self._get_array()
            self.y_pred,self.reservation = self.y_pred[:,:,:-1],self.y_pred[:,:,-1]
            self.y_pred_label = np.argmax(self.y_pred,axis=2)
            #print((self.y_pred_label,self.y_pred,self.reservation,'model_out_put'))
            
        elif loss_type == 'SelectiveLoss':
            self.four_rate_dic = {str(i):{'tp':0,'fp':0,'tn':0,'fn':0} for i in range(num_class)}
            self.y_pred, self.y_select,self.y_aux,self.y_true = self._get_array() # (sample,batch,class)
            self.y_pred_label = np.argmax(self.y_pred,axis=2)
            print('validation?',self.y_pred_label,self.y_pred,self.y_select)
        
        


        self.coverage = coverage # only for gambler input a list of coverage
        
        #check if loss_type is Gambler or Selective
        self.metrics = {f"{i}_coverage_{j}": {'f1': 0, 'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0,'loss':self.ave_loss} for i in range(self.num_class) for j in self.coverage}
        
        
    def calculate_selected_metrics(self):
        if self.loss_type == 'GamblerLoss':
            self._calculate_gambler_metrics()
        elif self.loss_type == 'SelectiveLoss':
            self._calculate_selective_metrics()
        return self.metrics

    def get_four_rate(self):
        y_true_one_hot = np.eye(self.num_class)[self.y_true.reshape(-1)]
        y_pred_one_hot = np.eye(self.num_class)[self.y_pred_label.reshape(-1)]
        y_pred_one_hot_tensor = torch.tensor(y_pred_one_hot)
        y_true_one_hot_tensor = torch.tensor(y_true_one_hot)
        print('y_pred_one_hot',y_pred_one_hot_tensor)
        confu_matrix = get_confusion_matrix(y_pred_one_hot_tensor,y_true_one_hot_tensor)
        for i in range(self.num_class):
            self.four_rate_dic[str(i)]['tp'] += confu_matrix[:,i,0].sum().item()
            self.four_rate_dic[str(i)]['fp'] += confu_matrix[:,i,1].sum().item()
            self.four_rate_dic[str(i)]['tn'] += confu_matrix[:,i,2].sum().item() 
            self.four_rate_dic[str(i)]['fn'] += confu_matrix[:,i,3].sum().item()
        return self.four_rate_dic

    def _calculate_gambler_metrics(self):
        #for every coverage, calculate its corresponding metrics
        for i in range(self.num_class):
            for j in self.coverage:
                #get each calculation metric
                output_coverage, pred_label_coverage, true_label_coverage = self._gambler_selective_pred(j)
                #print(output_coverage.shape,j,'length')
                y_pred_coverage = output_coverage[:,:-1]
                true_binary = (true_label_coverage == i).astype(int)
                pred_binary = (pred_label_coverage == i).astype(int)
                #print(true_binary,pred_binary)
                self.metrics[f"{i}_coverage_{j}"]['f1'] = f1_score(true_binary, pred_binary)
                self.metrics[f"{i}_coverage_{j}"]['precision'] = precision_score(true_binary, pred_binary)
                self.metrics[f"{i}_coverage_{j}"]['recall'] = recall_score(true_binary, pred_binary)

                if len(np.unique(true_binary)) > 1:
                    self.metrics[f"{i}_coverage_{j}"]['auc'] = roc_auc_score(true_binary, np.max(y_pred_coverage[:,],axis=1)).reshape(-1)
                    #some probelems here for roc
                    print(roc_auc_score(true_binary, y_pred_coverage[:,i].reshape(-1)),'roc_auc_score',true_binary,y_pred_coverage[:,i].reshape(-1))
                    
                self.metrics[f"{i}_coverage_{j}"]['accuracy'] = accuracy_score(true_binary, pred_binary)

        return self.metrics
    

    def _calculate_selective_metrics(self):
        #first filter out selective samples
        #self.metrics = {str(i): {'f1': 0, 'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0,'loss':self.ave_loss} for i in range(self.num_class)}
        #get selected samples
        self.y_pred,self.y_select,self.y_aux,self.y_true,self.y_pred_label = self._selectivenet_pred()
        #get one hot
        self.y_true_one_hot = np.eye(self.num_class)[self.y_true.reshape(-1)]
        self.y_pred_one_hot = np.eye(self.num_class)[self.y_pred_label.reshape(-1)]
        #self.y_pred_label = np.argmax(self.y_pred,axis=1)
        #print(self._selectivenet_pred())
        for i in range(self.num_class):
            for j in self.coverage:
                true_binary = (self.y_true == i).astype(int)
                pred_binary = (self.y_pred_label == i).astype(int)
                
                self.metrics[f"{i}_coverage_{j}"]['f1'] = f1_score(true_binary, pred_binary)
                self.metrics[f"{i}_coverage_{j}"]['precision'] = precision_score(true_binary, pred_binary)
                self.metrics[f"{i}_coverage_{j}"]['recall'] = recall_score(true_binary, pred_binary)

                if len(np.unique(true_binary)) > 1:
                    self.metrics[f"{i}_coverage_{j}"]['auc'] = roc_auc_score(true_binary, self.y_pred[:,i]).reshape(-1) #use max softmax

                    
                self.metrics[f"{i}_coverage_{j}"]['accuracy'] = accuracy_score(true_binary, pred_binary)


        return self.metrics
        

    def _gambler_selective_pred(self,coverage_rate):
        #get the reservation
       # print(self.y_pred,'y_pred_shape')
        output, reservation = self.y_pred.reshape(-1,self.num_class), self.reservation.reshape(-1)
        #print(output)
        predictions = np.argmax(output,axis=1).reshape(-1)#shape : [Sample,pre_prob]
        coverage_rate = int(round(len(reservation)) * coverage_rate)
        #print(coverage_rate,'coverage_rate')
        #sorted by the reservation
        sort_index = np.argsort(reservation)

        #if coverage_rate is larger than the length of the reservation
        if coverage_rate >= len(reservation):
            coverage_rate -= 1

        output = output[sort_index,:][:coverage_rate]
        predictions = predictions[sort_index][:coverage_rate]
        true_labels = self.y_true[sort_index][:coverage_rate]
        reservation = reservation[sort_index][:coverage_rate]
        print(coverage_rate,f'coverage_rate6666, reservation order {reservation}')
        #cat with reservation
        output = np.concatenate((output, reservation[:, np.newaxis]), axis=1)
        print(output,'output after selection')
        return output,predictions,true_labels
    
    def _selectivenet_pred(self,threshold=0.7):
        #sort all output followed by selection output
        y_select = self.y_select.reshape(-1) # (Sample,Batch,1) --> (Sample*Batch)
        y_pred = self.y_pred.reshape(-1,self.num_class) # (Sample,Batch,Class) --> (Sample*Batch,Class)
        y_aux = self.y_aux.reshape(-1,self.num_class) # (Sample,Batch,Class) --> (Sample*Batch,Class)
        y_pred_label = self.y_pred_label.reshape(-1) # (Sample,Batch) --> (Sample*Batch)
        y_true = self.y_true.reshape(-1) # (Sample,Batch) --> (Sample*Batch)
        
        sort_index = np.argsort(y_select)

        y_pred = y_pred[sort_index,:]
        y_aux = y_aux[sort_index,:]
        y_select = y_select[sort_index]
        y_true = self.y_true[sort_index]
        y_pred_label = y_pred_label[sort_index]
        
        #get select output that is larger than threshold and 
        select_mask = y_select > threshold
        
        y_select_filtered = y_select[select_mask]
        y_pred_filtered = y_pred[select_mask, :]
        y_aux_filtered = y_aux[select_mask, :]
        y_true_filtered = y_true[select_mask]
        y_pred_label_filtered = y_pred_label[select_mask]



        return y_pred_filtered,y_select_filtered,y_aux_filtered,y_true_filtered,y_pred_label_filtered

    def _get_array(self):
        # get y_pred numpy with shape (Sample, Batch, Class)
        if self.loss_type == 'GamblerLoss':
            self.y_pred = np.stack([y.detach().cpu().numpy() for y in self.y_pred],axis=0)
            self.y_true = np.array(self.y_true)

            return self.y_pred,self.y_true
        
        elif self.loss_type == 'SelectiveLoss':
            #get y_pred, y_select, y_aux numpy with shape (Sample, Batch, Class)
            #print(self.y_pred,'y_pred_shape vali!')
            self.y_select = [y[1] for y in self.y_pred]
            self.y_aux = [y[2] for y in self.y_pred]
            self.y_pred = [y[0]for y in self.y_pred]#last unpack to keep the same variable name

            self.y_pred = np.stack([y.detach().cpu().numpy()  for y in self.y_pred],axis=0)
            self.y_select = np.stack([y.detach().cpu().numpy()  for y in self.y_select],axis=0)
            self.y_aux = np.stack([y.detach().cpu().numpy()  for y in self.y_aux],axis=0)
            self.y_true = np.array(self.y_true)

            return self.y_pred,self.y_select,self.y_aux,self.y_true



"""
class Metrics():
    def __init__(self,num_class=2,y_pred=None,y_true_label=None,targets=None):
        
        # args:
        #     y_pred: list of predicted tensor
        #     y_true_label: list of true labels
        #     targets: dicts of targets and their labels
        
        self.num_class = num_class
        #turn into (steps,batch,out_class) prob
        #[batch,sample,class]
        self.y_pred = np.stack([y.detach().cpu().numpy() for y in y_pred],axis=0)#Prob of samples
        self.four_rate_dic = {str(i):{'tp':0,'fp':0,'tn':0,'fn':0} for i in range(num_class)}
        self.y_true_label = np.array(y_true_label)
        self.y_pred_label = [torch.argmax(y_pre,dim=1).detach().cpu().numpy().tolist() for y_pre in y_pred]
        self.y_pred_label = [item for sublist in self.y_pred_label for item in sublist]

        self.y_pred_label = np.array(self.y_pred_label)
        self.y_pred_one_hot = torch.nn.functional.one_hot(torch.tensor(self.y_pred_label,dtype=torch.int64),num_classes=self.num_class)
        self.y_true_one_hot = torch.nn.functional.one_hot(torch.tensor(self.y_true_label.tolist(),dtype=torch.int64),num_classes=self.num_class)
        self.metrics_df = pd.DataFrame()

    def calculate_metrics(self):
        self.metrics = {str(i): {'f1': 0, 'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0} for i in range(self.num_class)}

        for i in range(self.num_class):
            #make target class as positive class
            true_binary = (self.y_true_label == i).astype(int)
            pred_binary = (self.y_pred_label == i).astype(int)

            self.metrics[str(i)]['f1'] = f1_score(true_binary, pred_binary)
            self.metrics[str(i)]['precision'] = precision_score(true_binary, pred_binary)
            self.metrics[str(i)]['recall'] = recall_score(true_binary, pred_binary)
            
            #if have more at least 0,1 in the true_binary
            if len(np.unique(true_binary)) > 1:
                self.metrics[str(i)]['auc'] = roc_auc_score(true_binary, self.y_pred[:,:,i].reshape(-1))#error here should be prob of class 1

            self.metrics[str(i)]['accuracy'] = accuracy_score(true_binary, pred_binary)

        return self.metrics



    def get_roc(self,average='weighted'):
        #return compute_roc_auc(self.y_pred_one_hot,self.y_true_one_hot,average)
        positive_class = self.num_class - 1
        y_pred_pos = self.y_pred[:,:,positive_class].reshape(-1)

        return roc_auc_score(self.y_true_one_hot[:,positive_class],y_pred_pos,average=average)

    def get_four_rate(self) -> tensor:
"""
        # args:
        #     y_pred: (B,C) one-hot tensor
        #     y_true: (B,C) one-hot tensor
"""
        confu_matrix = get_confusion_matrix(self.y_pred_one_hot,self.y_true_one_hot)
        #calculate tp,fp,tn,fn
        for i in range(self.num_class):
            self.four_rate_dic[str(i)]['tp'] += confu_matrix[:,i,0].sum()
            self.four_rate_dic[str(i)]['fp'] += confu_matrix[:,i,1].sum() 
            self.four_rate_dic[str(i)]['tn'] += confu_matrix[:,i,2].sum() 
            self.four_rate_dic[str(i)]['fn'] += confu_matrix[:,i,3].sum()
        return self.four_rate_dic
    
    def get_accuracy(self) -> float:
        """"""
        args:
            y_pred_label: list of predicted labels
            y_true_label: list of true labels
        """"""""
        accuracy = accuracy_score(self.y_pred_label,self.y_true_label)
        return accuracy
    
    def get_f1_score(self,average='binary') -> float:
        positive_class = self.num_class - 1
        y_pred_pos = self.y_pred[:,:,positive_class].reshape(-1)
        return f1_score(self.y_true_one_hot[:,positive_class],self.y_pred_one_hot[:,positive_class],average=average)
    

    def generate_metrics_df(self, epoch):
        # 
        metrics_data = []
        for class_id, class_metrics in self.metrics.items():
            data_row = {"epoch": epoch}  # addepoch
            data_row.update({"class_id": class_id})  # add class_id
            data_row.update(class_metrics)  # add metrics
            metrics_data.append(data_row)

        # Create new df
        new_df = pd.DataFrame(metrics_data)
    # Using concat instead of append
        #self.metrics_df = pd.concat([self.metrics_df, new_df], ignore_index=True)

        return new_df
    #for regression

class Metrics_regression:
    def __init__(self,y_pred,y_true) -> None:
        self.y_pred = np.stack([y.detach().cpu().numpy() for y in y_pred],axis=0)#Prob of samples
        self.y_pred = self.y_pred.reshape(-1)
        self.y_true = np.array(y_true)

    def calculate_metrics(self):
        # Calculate Mean Squared Error
        mse = mean_squared_error(self.y_true, self.y_pred)
        
        # Calculate Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Calculate Mean Absolute Error
        mae = mean_absolute_error(self.y_true, self.y_pred)
        
        # Calculate R^2 Score
        r2 = r2_score(self.y_true, self.y_pred)

        # Store the metrics in a dictionary
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        return self.metrics
    
    
    def generate_metrics_df(self, epoch):
        # 
        metrics_data = []
        data_row = {"epoch": epoch}
        for class_metrics,value in self.metrics.items():
              # addepoch
            data_row[class_metrics] = value  # add class_id  # add metrics
        metrics_data.append(data_row)

        # Create new df
        new_df = pd.DataFrame(metrics_data)
    # Using concat instead of append
        #self.metrics_df = pd.concat([self.metrics_df, new_df], ignore_index=True)

        return new_df
"""