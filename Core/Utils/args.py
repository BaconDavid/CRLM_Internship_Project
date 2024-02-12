import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='train',required=True,help='train/vali or test')
    parser.add_argument('--config_file',type=str,required=True,help='path to config file')
    parser.add_argument('--fold',type=str,required=True,help='which fold to train')
    args = parser.parse_args()
    return args