import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
sys.setrecursionlimit(10**6)
import threading
threading.stack_size(2**26)


class ImageLoad:
    def __init__(self,input_path):
        """
        The path should only contain images
        """
        self.input_path = input_path
        self.images_names = os.listdir(self.input_path)
        self.images_num = len(self.images_names)

    def image_load(self):
        images_sitk = {}
        for name in self.images_names:
            try:
                img = sitk.ReadImage(self.input_path + name)
            except Exception as e:
                raise NameError("There are wrong files in the dir!")
                print(e)
            images_sitk[name] = img
        return images_sitk
        


class Resampler:
    def __init__(self,images_sitk,out_path):
        self.images_sitk = images_sitk
        self.out_image = {}
        self.out_path = out_path
    
    def resample(self,label=False,out_spacing=[0.7421875, 0.7421875, 1.0]):
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        print('here!')

        for name,image in self.images_sitk.items():
            original_spacing = image.GetSpacing()
            new_size = [int(round(image.GetSize()[i] * original_spacing[i] / out_spacing[i])) for i in range(3)]
            #set resample parameters
            #print(resample.SetOutputOrigin,'resample version')
            resample.SetOutputOrigin(image.GetOrigin())
            #print(image.GetOrigin(),'after resample version')
            resample.SetSize((new_size))
           
            resample.SetOutputDirection(image.GetDirection())

            resample.SetTransform(sitk.Transform())
            resample.SetDefaultPixelValue(image.GetPixelIDValue())

            #set interpolator
            if args.label:
                print('resample label image!')
                resample.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                print('here!')
                resample.SetInterpolator(sitk.sitkBSpline)

            out_image = resample.Execute(image)
            self.image_save(out_image,name)
            print('out_image size!',out_image.GetSize(),name)

    def image_save(self,out_image,name):
        sitk.WriteImage(out_image, self.out_path + name)

if __name__ == "__main__":
# 在你的main函数或者代码入口部分，添加下面的代码
    parser = argparse.ArgumentParser(description='Resample NIFTI images.')

    # 添加参数
    parser.add_argument('--input_path', required=True, type=str, help='Directory containing input images.')
    parser.add_argument('--output_path', required=True, type=str, help='Directory to save output images.')
    parser.add_argument('--out_spacing', nargs=3, default=[0.7421875, 0.7421875, 1.0], type=float, help='Output spacing for resampling.')
    parser.add_argument('--interpolator', default='bspline', type=str, choices=['bspline', 'nearest'], help='Interpolator for resampling.')
    parser.add_argument('--label', default=False, help='Resample label images.',type=bool)

    args = parser.parse_args()

    # 根据输入的参数来设置Resampler类
    interpolator = sitk.sitkBSpline if args.interpolator == 'bspline' else sitk.sitkNearestNeighbor
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f'Create the {args.output_path} directory')

    image_load = ImageLoad(args.input_path)
    images_sitk = image_load.image_load()
    resampler = Resampler(images_sitk,args.output_path)

    resampler.resample(out_spacing=args.out_spacing)



     