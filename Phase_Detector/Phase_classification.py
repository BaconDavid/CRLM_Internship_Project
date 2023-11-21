import os
import pandas as pd
import nibabel as nib
import datetime
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import label_binarize
import einops # for ViT

from torchsummary import summary
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    RandZoom,
    Compose,
    RandRotate,
    RandFlip,
    RandGaussianNoise,
    ToTensor,
    Resize,
    Rand3DElastic,
    RandSpatialCrop
    )


#################################################################################
#                                Utils functions                                #
#################################################################################

def plot(train_loss_epoch_x_axis, epoch_loss_values, val_loss_epoch_x_axis, val_loss_values, path, current_epoch):
    """
    Generate and save three types of loss plots (train loss, test loss, and combined) as PNG images.
    Additionally, save the loss data and x-axis values as numpy arrays.

    Parameters:
        train_loss_epoch_x_axis (list or array): X-axis values for the train loss plot (epochs).
        epoch_loss_values (list or array): Train loss values corresponding to each epoch.
        val_loss_epoch_x_axis (list or array): X-axis values for the test loss plot (epochs).
        val_loss_values (list or array): Test loss values corresponding to each epoch.
        path (str): Directory path where the plots and numpy arrays will be saved.
        current_epoch (int): The current epoch number for which the plots are being generated.
    """

    plt.plot(train_loss_epoch_x_axis,epoch_loss_values, label='Train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Train loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/Train_loss.png')
    plt.clf()

    plt.plot(val_loss_epoch_x_axis , val_loss_values, label='Test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'test loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/test_loss.png')
    plt.clf()


    plt.plot(train_loss_epoch_x_axis,epoch_loss_values, label='Train loss')
    plt.plot(val_loss_epoch_x_axis , val_loss_values, label='Test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Train and test loss (epoch {current_epoch})')
    plt.legend()
    plt.savefig(path+'/combi_loss.png')
    plt.clf()

    # also save them as a npy file
    np.save(path+'/train_loss.npy', epoch_loss_values)
    np.save(path+'/val_loss.npy', val_loss_values)
    np.save(path+'/train_loss_epoch_x_axis.npy', train_loss_epoch_x_axis)
    np.save(path+'/val_loss_epoch_x_axis.npy', val_loss_epoch_x_axis)
def data_path_and_labels(data_root: str,  label_path: str, anon_id: str = 'Subject', label_name: str = 'Phase'):
    """
    Read in image paths, mask paths and labels from Excel file, given the root directory of the images, the file path of
    the Excel file, and the corresponding column names for anonymous IDs and the labels. Labels are converted from a
    percentage to a fraction.

    Args:
        data_root: Root directory of the images.
        mask_root: Root directory of the masks.
        label_path: File path of the Excel file containing labels.
        anon_id: Name of the column in the Excel file containing anonymous IDs. Default is 'AnonID'.
        label_name: Name of the column in the Excel file containing the labels. Default is 'dHGP'.

    Returns:
        A tuple containing numpy arrays of image paths, mask paths and labels.
    """
    # read in the labels from the csv file
    labels_df = pd.read_csv(label_path)
    print(label_path,'hey label path')
    print("fuck ",labels_df.shape)

    # get the names and labels from the DataFrame
    #names = labels_df[anon_id].values
    labels = labels_df[label_name].values
    image_names = labels_df['Experiment'].values

    # initialize lists for storing the file paths
    images = []
    masks = []
    label_list = []

    # loop over the files in the data directory
    for filename in os.listdir(data_root):
        # add the images data
        images.append(os.path.join(data_root, filename))

    # convert the labels to a numpy array go form % to fraction 
    labels = np.array(list(labels)) 
    # convert the lists to numpy arrays for convenience
    images = np.array(images)
    masks = np.array(masks)
    print('This is label num and images num',len(labels),len(images))

    # check that the number of images and labels match
    if len(images) != len(labels):
        raise Exception('[ERROR]: Number of images and labels do not match')

    # convert the labels to PyTorch tensors
    labels = torch.as_tensor(labels).float()

    return images, labels
def stratified_cross_val(images: np.ndarray, labels: np.ndarray, folds: int = 5, fold: int = 0, continuous: bool = False):
    """
    Perform stratified k-fold cross-validation on the given images and labels.
    
    Args:
        images (np.ndarray): A numpy array of shape (n_samples) representing path to the images.
        labels (np.ndarray): A numpy array of shape (n_samples,) representing the labels for each image.
        folds (int, optional): The number of folds to use in the cross-validation. Defaults to 5.
        fold (int, optional): The fold to use for testing. Must be between 0 and `folds-1`. Defaults to 0.
        continuous (bool, optional): Whether the labels are continuous values (True) or discrete integers (False). Defaults to True.
    
    Returns:
        Tuple[List[int], List[int]]: A tuple of two lists representing the indices of the training and testing samples for the specified fold.
    """
    # Discretize continuous labels into integer bins
    if continuous == True:
        num_classes = 10
        label_bins = np.linspace(0, 1, num_classes + 1) # create equal-width bins for the labels
        discretized_labels = np.digitize(labels, label_bins) - 1 # convert labels to integers
    elif continuous == False:
        discretized_labels = labels

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=folds, random_state=0, shuffle=True)
    skf.get_n_splits(images, discretized_labels)
    train_idx_list = []
    test_idx_list = []
    
    
    
    for i, (train_index, test_index) in enumerate(skf.split(images, discretized_labels)):
        if i == fold:
            print(f"Fold {i+1}/{folds}")
            print("TRAIN:", len(train_index), "TEST:", len(test_index))

            train_idx_list.append(train_index)
            test_idx_list.append(test_index)

    return train_idx_list[0], test_idx_list[0]
def get_metrics(pred, label):
    '''
    Calculate relevant metrics for the given predictions and labels using sklearn.

    Args:
        pred: torch tensor of predictions.
        label: torch tensor of labels.
    
    Returns:
        AUC, accuracy, F1, TN, TP, FN, FP: float values of the respective metrics.
    '''

    # convert the predictions/labels to a numpy array

    # print('pred', pred)
    # pred = np.concatenate(pred, axis=0)
    # label = np.concatenate(label, axis=0)

    pred = [arrary.reshape(3) for arrary in pred]
    # calculate the AUC for 3 classes
    y_true_binarized = label_binarize(label, classes=[0, 1, 2])
    pred_label = [np.argmax(array) for array in pred]
# 计算多分类AUC
    AUC = roc_auc_score(y_true_binarized, pred, multi_class='ovr')

    print("AUC:", AUC)


    # convert the predictions to binary values
    
    # calculate the accuracy
    accuracy = accuracy_score(label, pred_label)
    macfro_F1 = f1_score(label, pred_label,average='macro')
    #calculate tn,tp,fn,fp
    conf_matrix = confusion_matrix(label, pred_label)
    num_classes = conf_matrix.shape[0]
    metrics = {}

    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        tn = conf_matrix.sum() - (fp + fn + tp)

        metrics[f'Class {i}'] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

    # return the metrics
    return AUC, accuracy, macfro_F1,metrics
def balanced_sampler(labels):
    '''
    Make a samples that is balanced between the classes. can be passed as an argument to the dataloader.

    args:
        labels: torch tensor of labels, of the training samples.
    returns:
        sampler: a balanced sampler.
    '''
    # to np array
    labels = labels.numpy().astype(int)
    # get the class distribution
    Blanco_labels_num = len(np.where(labels == 0)[0])
    AP_labels_num = len(np.where(labels == 1)[0])
    PVP_labels_num = len(np.where(labels == 2)[0])
    class_freq = [Blanco_labels_num, AP_labels_num, PVP_labels_num]
    # calculate the weights for each class

    weights = [1.0/class_freq[label] for label in labels]  # weights for each sample in the dataset
  
    return WeightedRandomSampler(weights, len(weights),replacement=True)


print('torch.cuda.is_available(): ',torch.cuda.is_available())
print('torch.backends.cudnn.enabled',torch.backends.cudnn.enabled)
print('torch.cuda.device_count()', torch.cuda.device_count())

# check if GPU is available
pin_memory = False
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

# make a directory to store the results, for this run
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_').replace('-', '_').replace(':', '.')
path = '.' + '/run_classify'+now_str
os.makedirs(path, exist_ok=True)
print(f"[INFO]: results saved to {path}]")

#################################################################################
#                                Network parameters                             #
#################################################################################

# cross validation parameters
folds = 5 
fold = 4

# data parameters
batch_size = 1
in_shape = (250,275,40)  #(250, 275, 40)
split = 0.2
data_aug = True
balanced_sampling = False
use_mask = False # if true set in_channels to 2, else to 1

# model parameters 
model_name = 'resnet10'
loss_function = 'BCE' 
learning_rate = 0.001 #1e-3
max_epochs = 200
experiment_name = 'TB OS LDA'
in_channels = 1 if use_mask == False else 2
model_eval = False

# validation parameters
val_interval = 1
plot_interval = 10
viusalize_train = False

print(f"[INFO]: Experiment name: {experiment_name} fold {fold+1}/{folds}")
print(f"[INFO]: saving to {path} ")

# set the paths to the data and masks
# data_root = "/data/scratch/r098986/CT_Phase/Data/Train_Data/"
# mask_root = './Test_Data/Seg_Phase_Data/'
# label_path = '/data/scratch/r098986/CT_Phase/Data/True_Label/FINAL_PHASE_LABEL.csv'
data_root = '../Test_Data/Train_Data/'
mask_root = '../Test_Data/Seg_Phase_Data/'
label_path = '../Test_Data/True_Label/FINAL_PHASE_LABEL.csv'
model_info = {
    'model_type':'classification',
    'data_aug':data_aug,
    "model_name": model_name,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": max_epochs,
    "optimizer": "Adam",
    "loss_function": loss_function,
    "dataset": data_root,
    "split": split,
    "experiment_name": experiment_name,
    "balanced_sampling": balanced_sampling,
    "use_mask": use_mask,
    "folds": folds,
    "fold": fold,
    'model in eval mode': model_eval,
    }

# save the parameters, as a json file
with open(path+'/model_info.json', 'w') as fp:
    json.dump(model_info, fp, indent=4)

#################################################################################
#                                Get the labels                                 #
#################################################################################

images, labels = data_path_and_labels(data_root,  label_path, anon_id='Experiment', label_name='Phase')


############# make a stratified split of the data #################

train_idx, test_idx = stratified_cross_val(images, labels, folds=folds, fold=fold, continuous=False)

#################################################################################
#                                make data loaders                              #
#################################################################################

# load a single image to get the shape
im = nib.load(images[0])
im = im.get_fdata()
print('im.shape: ', im.shape)

# get shape mask
#mask = nib.load(masks[0])
#mask = mask.get_fdata()
#print('mask.shape: ', mask.shape)
#print(f'mask values: {np.unique(mask)}')

# Define transforms
if data_aug == False:
    train_transforms = Compose([EnsureChannelFirst(), 
                                Resize(in_shape)
                                ])
elif data_aug == True:
    print('Data augmentation is on LDA')
    train_transforms = Compose([
                                EnsureChannelFirst(),
                                # Data augmentation
                                RandZoom(prob = 0.5, min_zoom=1.0, max_zoom=1.2),
                                RandRotate(range_z = 0.35, prob = 0.8),
                                RandFlip(prob = 0.5),
                                RandSpatialCrop((186,144,75), random_size=False),
                                # To tensor
                                ToTensor()
                                ])

val_transforms = Compose([EnsureChannelFirst(), 
                          RandSpatialCrop((186,144,75),random_size=False),
                                ])

# # Define nifti dataset, data loader
check_ds = ImageDataset(image_files=images, seg_files=None, labels=labels, transform=train_transforms, 
                        seg_transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)

sampler = balanced_sampler(labels[train_idx])

# create a training data loader
train_ds = ImageDataset(image_files=images[train_idx], 
                        labels=labels[train_idx], transform=train_transforms, seg_transform=train_transforms)

if balanced_sampling == True:
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory, sampler=sampler)
elif balanced_sampling == False:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=images[test_idx], labels=labels[test_idx], 
                      transform=val_transforms,seg_transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=pin_memory)


print(f'mean test set: {torch.mean(labels[test_idx])}, min test set: {torch.min(labels[test_idx])}, max test set: {torch.max(labels[test_idx])}')
print(f'mean train set: {torch.mean(labels[train_idx])}, min train set: {torch.min(labels[train_idx])}, max train set: {torch.max(labels[train_idx])}')


#################################################################################
#                                  model                                        #
#################################################################################

# list to store metrics
epoch_loss_values, train_loss_epoch_x_axis = [], []
val_loss_values, val_loss_epoch_x_axis = [], []
best_val_loss = np.inf

# define the model
model = monai.networks.nets.resnet10(n_input_channels=in_channels, num_classes=3, widen_factor=1).to(device)

if loss_function == 'BCE':
    loss_function = torch.nn.CrossEntropyLoss()
else:
    print('NO loss function defined!')

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

sigmiod = torch.nn.Sigmoid()

# init a pandas dataframe to store the results
train_results_df = pd.DataFrame(columns=[
    'epoch', 'train_loss', 'train_AUC', 'train_accuracy', 'train_F1', 
    'train_TN_blanco', 'train_TP_blanco', 'train_FN_blanco', 'train_FP_blanco',
    'train_TN_AP', 'train_TP_AP', 'train_FN_AP', 'train_FP_AP',
    'train_TN_PVP', 'train_TP_PVP', 'train_FN_PVP', 'train_FP_PVP'
])
#train_results_df = pd.DataFrame(columns=['epoch', 'train_loss','train_AUC', 'train_accuracy', 'train_F1', 'train_TN', 'train_TP', 'train_FN', 'train_FP' ])
test_results_df = pd.DataFrame(columns=[
    'epoch', 'train_loss', 'test_AUC', 'test_accuracy', 'test_F1', 
    'test_TN_blanco', 'test_TP_blanco', 'test_FN_blanco', 'test_FP_blanco',
    'test_TN_AP', 'test_TP_AP', 'test_FN_AP', 'test_FP_AP',
    'test_TN_PVP', 'test_TP_PVP', 'test_FN_PVP', 'test_FP_PVP'
])
#test_results_df = pd.DataFrame(columns=['epoch', 'test_loss','test_AUC', 'test_accuracy', 'test_F1', 'test_TN', 'test_TP', 'test_FN', 'test_FP'])

#################################################################################
#                                train  model                                   #
#################################################################################

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    print("-" * 10)

    model.train() # set the model to training mode

    # init loss
    epoch_loss = 0
    val_loss = 0
    step = 0
    
    # init output and lstart = time.time()abel list for metrics
    train_label = []
    train_pred = []
    train_pred_raw = []

    # loop over the training data
    for im,label in train_loader:
        step += 1
        
        # clear the gradients
        optimizer.zero_grad()
        
        # move the data to the device
        input, label = im.to(device),label.to(device)

        if use_mask == True:
            # stack the input and seg together
            input = torch.cat((input,seg), dim=1)
        if viusalize_train == True:
            # plot to check images are correct
            print('image shape: ', input.shape)
            input_np = input.detach().cpu().numpy()
            print('input_np shape: ', input_np.shape)
            plt.imshow(input_np[0,0,:,:,20],vmin = 30, vmax=150, cmap='gray')
            plt.title((label))
            plt.show()
            plt.savefig(os.path.join(path, 'image'+str(step)+'.png'))
            plt.clf()
            
        
        # forward pass
        output = (model(input))        
        train_pred_raw.append(output) # save the pre-sigmiod output

        output = sigmiod(output) 
        loss = loss_function(output, label.long())

        train_label.append(label)
        print('this is label',label)
        train_pred.append(output)# save the post-sigmiod output
        
        loss.backward() # backward pass
        optimizer.step() # update the model weights


        epoch_loss += loss.item() # accumulate the loss, to calculate the average loss at the end of the epoch
        epoch_len = len(train_ds) // train_loader.batch_size # calculate the number of steps in an epoch
    print("this is train label",train_label)
    ########## log training performance ##########

    # log the loss
    epoch_loss = epoch_loss / step # average loss, over the epoch
    epoch_loss_values.append(epoch_loss)
    train_loss_epoch_x_axis.append(epoch+1)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # log the metrics
    if (epoch+1) % val_interval == 0:
        # calculate the metrics
        AUC, accuracy, F1, conf_matrix_dic = get_metrics([array.detach().cpu().numpy() for array in train_pred], 
                                                        [array.detach().cpu().numpy().item() for array in train_label])
        print("train_pred",train_pred)
        # write metrics to df 
        new_row = {'epoch': epoch+1, 'train_loss': epoch_loss, 'train_AUC': AUC, 'train_accuracy': accuracy, 'train_F1': F1,
                'train_TN': tn, 'train_TP': tp, 'train_FN': fn, 'train_FP': fp}
        new_row = {'epoch': epoch+1, 'train_loss': epoch_loss, 'train_AUC': AUC, 'train_accuracy': accuracy, 'train_F1': F1,
        'train_TN_blanco': conf_matrix_dic['Class 0']['TN'], 'train_TP_blanco': conf_matrix_dic['Class 0']['TP'], 'train_FN_blanco': conf_matrix_dic['Class 0']['FN'], 'train_FP_blanco': conf_matrix_dic['Class 0']['FP'],
        'train_TN_AP': conf_matrix_dic['Class 1']['TN'], 'train_TP_AP': conf_matrix_dic['Class 1']['TP'], 'train_FN_AP': conf_matrix_dic['Class 1']['FN'], 'train_FP_AP': conf_matrix_dic['Class 1']['FP'],
        'train_TN_PVP': conf_matrix_dic['Class 2']['TN'], 'train_TP_PVP': conf_matrix_dic['Class 2']['TP'], 'train_FN_PVP': conf_matrix_dic['Class 2']['FN'], 'train_FP_PVP': conf_matrix_dic['Class 2']['FP']
        }
        train_results_df = pd.concat([train_results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

#################################################################################
#                                Test  Model                                    #
#################################################################################

    val_step = 0
    val_label = []
    val_pred = []
    val_pred_raw = []

    if (epoch+1) % val_interval == 0:
        
        if model_eval:
            model.eval() # set in eval mode
        
        # loop over the validation data
        for im, label in val_loader:
            val_step += 1
            # move the data to the device
            input,label = im.to(device), label.to(device)
            # stack the input and seg together

            if use_mask == True:
                input = torch.cat((input,seg), dim=1)
            
            # test forward pass, so dont track gradients
            with torch.no_grad():
                output = (model(input))
                if model_name == 'ViT':
                    output =  output[0] # get the output from the ViT model
                val_pred_raw.append(output)
                output = sigmiod(output)
                loss = loss_function(output, label.long())
                # keep tack of the labels and predictions
                val_label.append(label)
                val_pred.append(output)
                val_loss += loss.item()
                val_epoch_len = len(val_ds) // val_loader.batch_size # calculate the number of steps in an epoch

        # get average test loss
        val_loss = val_loss / val_step
        val_loss_values.append(val_loss)
        val_loss_epoch_x_axis.append(epoch+1)

        # calculate the metrics
        AUC, accuracy, F1, conf_matrix_dic = get_metrics([array.detach().cpu().numpy() for array in val_pred], 
                                                        [array.detach().cpu().numpy().item() for array in val_label])
        # write metrics to df 
        new_row = {'epoch': epoch+1, 'test_loss': val_loss, 'test_AUC': AUC, 'test_accuracy': accuracy, 'test_F1': F1,
                'test_TN_blanco': conf_matrix_dic['Class 0']['TN'], 'test_TP_blanco': conf_matrix_dic['Class 0']['TP'], 'test_FN_blanco': conf_matrix_dic['Class 0']['FN'], 'test_FP_blanco': conf_matrix_dic['Class 0']['FP'],
                'test_TN_AP': conf_matrix_dic['Class 1']['TN'], 'test_TP_AP': conf_matrix_dic['Class 1']['TP'], 'test_FN_AP': conf_matrix_dic['Class 1']['FN'], 'test_FP_AP': conf_matrix_dic['Class 1']['FP'],
                'test_TN_PVP': conf_matrix_dic['Class 2']['TN'], 'test_TP_PVP': conf_matrix_dic['Class 2']['TP'], 'test_FN_PVP': conf_matrix_dic['Class 2']['FN'], 'test_FP_PVP': conf_matrix_dic['Class 2']['FP']
                }
        test_results_df = pd.concat([test_results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        # save the datframes
        train_results_df.to_csv(path+'/train_results_df.csv')
        test_results_df.to_csv(path+'/test_results_df.csv')


        if (epoch+1) % plot_interval == 0:
            print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")
            plot(train_loss_epoch_x_axis, epoch_loss_values, val_loss_epoch_x_axis, val_loss_values, path, (epoch+1))

        # save best performing model
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_loss_epoch = epoch + 1
            # save the labels and predictions
            np.save(path+'/val_label_best.npy', [array.detach().cpu().numpy().item() for array in val_label])
            np.save(path+'/val_pred_best.npy', [array.detach().cpu().numpy() for array in val_pred])
            torch.save(model.state_dict(), "best_metric_model_regression3d_array.pth")
            print("saved new best metric model")

            print(f"Current epoch: {epoch+1} current best_val_loss: {best_val_loss:.4f} ")
            print(f"best_val_loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}")

# save the labels and predictions of the test set 
np.save(path+'/val_label_last.npy', [array.detach().cpu().numpy().item() for array in val_label])
np.save(path+'/val_pred_last.npy', [array.detach().cpu().numpy() for array in val_pred])


#################################################################################
#                                Saving results                                 #
#################################################################################

plot(train_loss_epoch_x_axis, epoch_loss_values, val_loss_epoch_x_axis, val_loss_values, path, max_epochs)
print(path)