import torch
import os
import sys




def path_check(func):
    def wrapper(*args,**kwargs):
        if not os.path.exists(args[2]):
            os.makedirs(args[2])
            print(f'Create the {args[2]} directory')
        return func(*args,**kwargs)
    return wrapper


@path_check
def save_checkpoint(save_path,state_dict,file_name):
    """
    args:
        state_dict: model parameters
        save_path: file save path
        file_name: file save name
    """

    save_name = os.path.join(save_path,file_name)
    torch.save(state_dict,save_name)
    current_epoch = state_dict['epoch']
    print("save model to {}".format(save_name))
    print("current epoch is {}".format(current_epoch))


def load_checkpoint(cfg,model,optimizer,scheduler,checkpoint_path):
    """
    args:
        model: model to be trained
        optimizer: optimizer
        scheduler: scheduler
        checkpoint_path: path to checkpoint
    """
    if os.path.exists(checkpoint_path):
        print("loading checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("loading checkpoint successfully")
        return checkpoint['epoch']
    else:
        print("no checkpoint found at {}".format(checkpoint_path))
        return 0