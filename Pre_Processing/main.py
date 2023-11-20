import argparse
import Crop,Resample,Xnat_Download,Data_Check

def run_resample(input_dir, output_dir, out_spacing=[0.7421875, 0.7421875, 1.0], **kwargs):
    parameters = {"out_spacing": out_spacing}

    for key, value in kwargs.items():
        parameters[key] = value

    image_load = Resample.ImageLoad(input_dir)
    images_sitk = image_load.image_load()
    resampler = Resample.Resampler(images_sitk)
    resampler.resample()
    resampler.image_save(output_dir)

def main():
    parser = argparse.ArgumentParser(description="Preprocessing CT images")
    tasks = {"resample": run_resample}

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



