import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help='path to the data folder')
    parser.add_argument('--label_path',type=str,help='path to the label csv')
    parser.add_argument('--epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--mode',type=str,default='train',help='train/vali or test')
    #parser.add_argument('--model_save_path',type=str,help='path to save model')
    #parser.add_argument('--visual_out_path',type=str,help='path to save visualized images')
    parser.add_argument('--result_save_path',type=str,help='path to save results')
    parser.add_argument('--device',type=str,default='cpu',help='device to train on')
    parser.add_argument('--Debug',type=str,default='False',help='Use a small subset of dataset to debug')
    args = parser.parse_args()
    return args