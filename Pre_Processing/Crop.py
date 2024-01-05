###### Liver Bounding Box ######
# This script is used to generate liver bounding box for each patient
# The bounding box is used to crop the liver region from the original image

import os
import nibabel as nib
import numpy as np
import skimage 
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib



class ImageLoad:
    def __init__(self,input_path):
        """
        The path should only contain images
        """
        self.input_path = input_path


        #self.images_names = os.listdir('../../Test_Data/Test_3D/')
        self.images_names = os.listdir(self.input_path)
        self.images_num = len(self.images_names)
        self.image_path = [self.input_path + name for name in self.images_names]

    def image_load(self,image_path,reader='nib'):
        if reader == 'nib':
            return nib.load(image_path)
        elif reader == 'sitk':
            return sitk.ReadImage(image_path)
        else:
            raise ValueError("The reader should be either nib or sitk!")
        
    #def image_get_float(self):

  


class LiverBoundingBox:
    def __init__(self,liver_mask,liver_orig,out_path_box=None,file_name=None):
        """
        args:
            liver_mask: the liver mask|nib or sitk
            original_image: the original image|nib or sitk
        """
        self.liver_seg = liver_mask
        self.liver_orig = liver_orig
        self.out_path_box = out_path_box
        self.file_name = file_name
        # self.original_image = original_image
        # self.liver_mask_array = self._get_array(liver_seg)
        # self.original_arrary = self._get_array(original_image)
        # self.image_loader = image_loader
    def extract_liver(self,liver=True):
        """
            get either the liver or tumor region from the mask
            args:
                mask: mask of the liver and tumor
                liver: if True return the liver region, if False return the tumor region
        """
        mask = self._get_array(self.liver_seg)
        mask = mask.astype(int)
        print(mask.shape)
        
        if liver:
            mask[mask == 2] = 1
        else:
            mask[mask == 1] = 0
            mask[mask == 2] = 1
            mask = mask.astype(int)
            
        # only keep the largest connected component
        labeled = skimage.measure.label(mask, connectivity=2)

        labeled[labeled != 1] = 0
      
        mask = labeled
        print(mask.sum())

        return mask


        
    def get_liver_bounding_box(self,liver_mask):
        '''
    Function to generate bounding box for liver, from a binary liver mask
        args:
            liver_mask: binary mask of the liver

        returns:
            bbox: bounding box of the liver (min_row, min_col, min_slice, max_row, max_col, max_slice = bbox)
        '''

        # get the image_probs
        image_probs = skimage.measure.regionprops((liver_mask))

        # get the bounding box of the liver

        if len(image_probs) == 0:
            print(f'[WARNING] no liver found')
            self._recording_failing()
            return None

        ## find the adjacent box that contains the liver
        for props in image_probs:
            bbox = props.bbox
            min_row, min_col, min_slice, max_row, max_col, max_slice = bbox 
            print("this is range",min_row, min_col, min_slice, max_row, max_col, max_slice)
        return [min_row, min_col, min_slice, max_row, max_col, max_slice]

    def crop_scan(self,liver_bounding):
        """
        Crop the scan with the bounding box
        args:
            liver_bounding: the bounding box of the liver
        """
        liver_original = self._get_array(self.liver_orig)
        # get the bounding box of the liver
        min_row, min_col, min_slice, max_row, max_col, max_slice = list(map(self._check_range,liver_bounding))
        print('this is after check range',min_row, min_col, min_slice, max_row, max_col, max_slice)
        
        #crop the scan
        cropped_scan = liver_original[min_row:max_row,min_col:max_col,min_slice:max_slice]
        return cropped_scan
    
    def store_cropped_data(self,cropped_data):

        if not os.path.exists(self.out_path_box):
            os.mkdir(self.out_path_box)
            print('The path does not exist, create the path!')

        header = self.liver_orig.header
        affine = self.liver_orig.affine
        print("this is shape of cropped data",cropped_data.shape)
        cropped_image = nib.Nifti1Image(cropped_data, affine, header)


        
        nib.save(cropped_image, self.out_path_box + self.file_name)

    @staticmethod   
    def liver_detection(mask):
        if np.count_nonzero(mask) != 0:
            return True


    def _check_range(self,range_num):
        return max(0,range_num)
    
    def _get_array(self,image_file):
        return image_file.get_fdata()
    
    def _recording_failing(self):
        with open(self.out_path_box + 'failing_box.txt','a') as f:
            f.write(self.file_name + '\n')
if __name__ == "__main__":

    # load the image
    #image_orig_path = '/data/scratch/r098986/CT_Phase/Data/Raw_Phase_data/'
    #image_seg_path = '/data/scratch/r098986/nnUnet_Seg/nnUNeT_test/Task_502_Phase_Data/'
    #cropped_out_path = '/data/scratch/r098986/CT_Phase/Data/Cropped_Data/'
    image_orig_path = 'D:/Onedrive/bioinformatics_textbook/VU_Study/internship/Erasmusmc/Test_Data/Raw_Phase_data/'
    image_seg_path = 'D:/Onedrive/bioinformatics_textbook/VU_Study/internship/Erasmusmc/Test_Data/Seg_Phase_data/'
    cropped_out_path = '../../Test_Data/CT_Phase/Cropped_Data_Liver/'
    #load the images
    image_orig_load = ImageLoad(image_orig_path)
    image_seg_load = ImageLoad(image_seg_path)
    for i in range(image_orig_load.images_num):
        file_name = image_orig_load.images_names[i]
        image_orign,image_segg = image_orig_load.image_load(image_orig_load.image_path[i]),image_seg_load.image_load(image_seg_load.image_path[i])

        #finding bounding box
        liver_bbox = LiverBoundingBox(image_segg,image_orign,cropped_out_path,file_name)
        liver_mask = liver_bbox.extract_liver()
        if liver_bbox.liver_detection(liver_mask):
            liver_box_range = liver_bbox.get_liver_bounding_box(liver_mask)
            cropped_image = liver_bbox.crop_scan(liver_box_range)
            liver_bbox.store_cropped_data(cropped_image)
        else:
            liver_bbox._recording_failing()




    # image_orig_nib,image_seg_nib = image_orig_load.images_nib,image_seg_load.images_nib
    # #get the images
    # for orig,seg in zip(image_orig_nib.items(),image_seg_nib.items()):
    # #get the liver mask
    # #create the liver bounding box
    #     orig_name,seg_name = orig[0],seg[0]
    #     orig_nib,seg_nib = orig[1],seg[1]
    #     liver_bbox = LiverBoundingBox(original_image=orig_nib,liver_seg=seg_nib)
    #     # get the bounding box of the liver
    #     liver_mask = liver_bbox.extract_liver()
    #     print(liver_mask.sum(),'hhhhhhhhhhh')
    #     if liver_bbox.liver_detection(liver_mask):
    #         liver_bounding = liver_bbox.get_liver_bounding_box(liver_mask)
    #         cropped_image = liver_bbox.crop_scan(liver_bounding)
    #         liver_bbox.store_cropped_data(cropped_image,'./Test_Data/Cropped_Data/',orig_name)
    #         print(f"The file {orig_name} has liver! and the shape is {cropped_image.shape}")
    #     else:
    #         print(f"The file {orig_name} has no liver!")
    #         with open('./Test_Data/Cropped_Data/No_Liver.txt','a') as f:
    #             f.write(orig_name + '\n')

       
        # check if the mask is empty
        

        
        #print(cropped_image)
    #mask_path = '/home/weiweiduan/Downloads/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task50_CILM/labelsTr/'
    #image_name = 'CILM_1_0000.nii.gz'
    #mask_name = 'CILM_1_0000.nii.gz'
    # create the liver bounding box
    #liver_bbox = LiverBoundingBox(mask_data,image_data)
    # get the liver region
    #liver_mask = liver_bbox.extract_liver()
    # get the bounding box of the liver
    #liver_bounding = liver_bbox.get_liver_bounding_box(liver_mask)
    # crop the image
    # cropped_image = liver_bbox.crop_scan(liver_bounding)
    # # plot the image
    # plt.imshow(cropped_image[:,:,0])
    # plt.show()
    # # save the image
    # cropped_image_nifti = nib.Nifti1Image(cropped_image, image.affine, image.header