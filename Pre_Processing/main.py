import argparse
import Crop,Resample,Xnat_Download,Data_Check
from Crop import ImageLoad


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing CT images")
    parser.add_argument('--data_dir',type=str,help='path to the data folder')
    parser.add_argument('--mask_dir',type=str,help='path to the mask_dir')
    parser.add_argument('--output_dir',type=str,help='path to the output folder')
    parser.add_argument('--window_pars',type=list,required=False,help='window level and window width')
    parser.add_argument('--resample_spacing',type=list,required=True,help='resample size')

    args = parser.parse_args()
    return args




def main():
    parser = argparse.ArgumentParser(description="Preprocessing CT images")

    subparsers = parser.add_subparsers(dest="Task", help="Sub-commands help")
    parser_resample = subparsers.add_parser("resample", help="Resample help")
    parser_resample.add_argument("--input_file", dest="input_file", type=str, help="input path")
    parser_resample.add_argument("--output_file", dest="output_file", type=str, help="output path")
    
    args = parser.parse_args()
    if args.Task in tasks:
        tasks[args.Task](args.input_file, args.output_file)
    print(args.input_file, args.output_file)

if __name__ == "__main__":
    main()



