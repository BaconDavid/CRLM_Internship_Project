import matplotlib.pyplot as plt
import numpy as np


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