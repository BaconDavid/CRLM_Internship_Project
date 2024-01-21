import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default='train',required=True,help='train/vali or test')
    parser.add_argument('--config_file',type=str,required=True,help='path to config file')
    args = parser.parse_args()
    return args