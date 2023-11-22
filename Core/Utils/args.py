import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,help='path to the data folder')
    parser.add_argument('--epochs',type=int,help='number of epochs')
    parser.add_argument('--mode',type=str,help='train/vali or test')
    parser.add_argument('--model_save_path',type=str,help='path to save model')
    parser.add_argument('--visual_out_path',type=str,help='path to save visualized images')
    parser.add_argument('--result_save_path',type=str,help='path to save results')
    args = parser.parse_args()
    return args