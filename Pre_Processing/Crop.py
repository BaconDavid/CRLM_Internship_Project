###### Liver Bounding Box ######
# This script is used to generate liver bounding box for each patient
# The bounding box is used to crop the liver region from the original image

import os
from random import choices
import nibabel as nib
import numpy as np
import skimage 
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import pandas as pd


def components_to_array(components):
    #
    component_shapes = list(components.values())[0].shape
    
    num_components = len(components)
    new_array = np.zeros((num_components,) + component_shapes, dtype=int)

    for i, (key, value) in enumerate(components.items()):
        new_array[i] = value

    return new_array
class FileLoad:
    def __init__(self,scans_file):
        """
        Load scans info from the csv file
        """
        self.scans_file = pd.read_csv(scans_file)
    
    @property
    def scans_df(self):
        return self.scans_file

class ImageSet:
    def __init__(self, image_path):
        """
        Initialize with the directory path containing images.
        """
        self.image_path = image_path
        self.image_names = os.listdir(self.image_path)
        self.image_num = len(self.image_names)
        self.image_full_paths = [os.path.join(self.image_path, name) for name in self.image_names]
        print("Total number of images:", self.image_num)

    def get_image_list(self):
        return self.image_names
    
    def get_image_num(self):
        return self.image_num
    
    def get_image_full_path(self, index):
        if index < 0 or index >= self.image_num:
            raise IndexError("Index out of range")
        
        return self.image_full_paths[index]



class ImageReader:
    def __init__(self,image_set):
        self.image_set = image_set
        self.current_index = 0

    def load_image(self,reader='sitk'):
        if self.current_index >= self.image_set.get_image_num():
            raise StopIteration("No more images to read")
        image_path = self.image_set.get_image_full_path(self.current_index)

        if reader == 'nib':
            image_array = nib.load(image_path).get_fdata()
        elif reader == 'sitk':
            #get image array
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
        else:
            raise ValueError("The reader should be either nib or sitk!")
        self.current_index += 1
        return image_array

    def load_image_name(self):
        if self.current_index >= self.image_set.get_image_num():
            raise StopIteration("No more images to read")
        image_name = self.image_set.get_image_list()[self.current_index]
        self.current_index += 1
        return image_name
class BoundingBoxRecorde:
    def __init__(self,file_name):
        self.file_name = file_name

    def Largest_Tumor(self):
        pass


class LiverBoundingBox():
    def __init__(self, liver_img, liver_mask: np.ndarray=None):
        """
        args:
            liver_img: the liver image array
            liver_mask: the liver mask array
        """
        self.liver_mask = liver_mask
        self.liver_img = liver_img

    def extract_liver(self,largest: bool = True):
        mask = self.liver_mask.astype(int)
        print(mask.shape)
        mask[mask == 2] = 0 #only keep liver regions
        labeled,num_features = skimage.measure.label(mask, connectivity=2,return_num=True)
        #only extract largest connected component
        if largest:
            max_size = 0
            for label in range(1, num_features + 1):
                label_size = np.sum(labeled == label)
                if label_size > max_size:
                    max_label = label
                    max_size = label_size
            mask = np.where(labeled == max_label, 1, 0)
        return mask
    

    def get_liver_bounding_box(self):
        image_probs = skimage.measure.regionprops((self.liver_mask))

        # get the bounding box of the liver

        if len(image_probs) == 0:
            print(f'[WARNING] {self.file_name} no liver found')

        ## find the adjacent box that contains the liver
        for props in image_probs:
            bbox = props.bbox
            print(bbox)
            min_slice, min_col, min_row, max_slice, max_col, max_row = bbox 
            print("this is range",min_slice, min_col, min_row, max_slice, max_col, max_row)
        return [min_slice, min_col, min_row, max_slice, max_col, max_row]
    
    def __check_liver_mask(self):
        pass



class TumorBoundingBox():
    def __init__(self, tumor_seg_mask,file_name=None,largest=True):
        self.seg_mask = tumor_seg_mask
        self.file_name = file_name
        self.tumor_mask = self.extract(largest)
        self.largest = largest
        

    def extract(self,largest=True):
        mask = self.seg_mask
        mask = mask.astype(int)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        if largest:
            mask = self.__get_largest_part(mask)
            return mask
        
        mask = mask.astype(int)

        return mask


    def get_bound_box(self,choice='pertumor'):
        choices = {'per_tumor':self.__get_per_tumor_bound_box(),}
                   #'all_tumor':self._get_all_tumor_bound_box,
                   #'largest_tumor':self._get_largest_tumor_bound_box}
        return choices[choice]

        

    def __get_per_tumor_bound_box(self):
        """"
        extract a bounding box of each tumor, return a np.arrary with shape (n,min_slice,min_col,min_row,max_slice,max_col,max_row,tumor_size) for each patient

        mask: mask that contains all tumors
        
        """
        if self.largest:
            raise ValueError("The mask is the largest tumor mask, cannot extract per tumor bounding box")
        per_tumor_mask_lst = []
        tumor_boduning_lst = []
        labeled,num_features = skimage.measure.label(self.tumor_mask, connectivity=2,return_num=True)

        for label in range(1, num_features + 1):
            component_mask = np.where(labeled == label, 1, 0)
            per_tumor_mask_lst.append(component_mask)
        
        #get tumor slices (n,d,h,w)
        per_tumor_mask_array = np.stack(per_tumor_mask_lst)
         # check each tumor
        for i in range(per_tumor_mask_array.shape[0]):  # how many slices
            tumor_slice = []
            # get the mask of each tumor
            tumor_mask_i = per_tumor_mask_array[i, :, :,:]
            # check each tumor slice
            for j in range(tumor_mask_i.shape[0]):
                slice = tumor_mask_i[j, :, :]
                if np.any(slice):
                    tumor_slice.append(j)
            # check tumor's bounding box
            
            ones_indices = np.argwhere(tumor_mask_i == 1)
            if ones_indices.size > 0:
                # max and min of the indices
                min_z, min_y, min_x = ones_indices.min(axis=0)
                max_z, max_y, max_x = ones_indices.max(axis=0)

            #calculate tumor size
            tumor_size = tumor_mask_i.sum()
            
            tumor_slice_lower = tumor_slice[0]
            tumor_slice_upper = tumor_slice[-1]
            tumor_boduning_lst.append((tumor_slice_lower, min_y,min_x,tumor_slice_upper,max_y,max_x,tumor_size))
            tumor_bounding_bbx_array = np.stack(tumor_boduning_lst)
        return tumor_bounding_bbx_array


    def __per_tumor_components(self):
        """
        args:mask: the mask of the tumor

        return: the mask of single tumor in a patient
        """
        #find each single connected component
        labeled,num_features = skimage.measure.label(self.mask, connectivity=2,return_num=True)
        #if only one connected component
        if num_features <= 1:
            return {1:labeled}
        # multiple connected components
        components = {}  

        for label in range(1, num_features + 1):
            component_mask = np.where(labeled == label, 1, 0)
            components[label] = component_mask
        return components


    def __get_largest_part(self,mask):
        max_label = 0
        max_size = 0
        labeled,num_features = skimage.measure.label(mask, connectivity=2,return_num=True)

        for label in range(1, num_features + 1):
            label_size = np.sum(labeled == label)
            if label_size > max_size:
                max_label = label
                max_size = label_size

        # keep only the largest connected component
        mask = np.where(labeled == max_label, 1, 0)
        mask.astype(int)
        return mask

    
    def __per_tumor_size(self):
        pertumor_components = self.__per_tumor_components()
        per_tumor_size = {}
        for key, value in pertumor_components.items():
            per_tumor_size[key] = value.sum()
        return per_tumor_size


    def __all_tumor_bound_box(self):
        pass
        
    
    def __largest_tumor_bound_box(self):
        pass

    def per_tumor_bound_box(self):
        per_tumor_size = self.__per_tumor_components()
        per_tumor_mask = components_to_array(per_tumor_size)
        bounding_box_dict = {'min_row':[],'min_col':[],'min_slice':[],'max_row':[],'max_col':[],'max_slice':[]}
        for key, value in per_tumor_mask.items():
            mask = value
            ones_indices = np.argwhere(mask == 1)
            if ones_indices.size > 0:
                # max and min of the indices
                min_z, min_y, min_x = ones_indices.min(axis=0)
                max_z, max_y, max_x = ones_indices.max(axis=0)

                # create the bounding box
                bounding_box = ((min_z, min_y, min_x), (max_z, max_y, max_x))
                bounding_box_dict['min_row'].append(min_x)
                bounding_box_dict['min_col'].append(min_y)
                bounding_box_dict['min_slice'].append(min_z)
                bounding_box_dict['max_row'].append(max_x)
                bounding_box_dict['max_col'].append(max_y)
                bounding_box_dict['max_slice'].append(max_z)


                print("Bounding box:", bounding_box)

        rows = []


        for key, values in bounding_box_dict.items():
            for i in range(len(values['min_row'])):
                row = {
                    'Sample': key,
                    'min_row': values['min_row'][i],
                    'min_col': values['min_col'][i],
                    'min_slice': values['min_slice'][i],
                    'max_row': values['max_row'][i],
                    'max_col': values['max_col'][i],
                    'max_slice': values['max_slice'][i]
                }
                rows.append(row)

        # 创建一个DataFrame
        df = pd.DataFrame(rows)

        return df


    

    

# class LiverBoundingBox:
#     def __init__(self,liver_mask,liver_orig,out_path_box=None,file_name=None):
#         """
#         args:
#             liver_mask: the liver mask|nib or sitk
#             original_image: the original image|nib or sitk
#         """
#         self.liver_seg = liver_mask
#         self.liver_orig = liver_orig
#         self.out_path_box = out_path_box
#         self.file_name = file_name
#         # self.original_image = original_image
#         # self.liver_mask_array = self._get_array(liver_seg)
#         # self.original_arrary = self._get_array(original_image)
#         # self.image_loader = image_loader
#     def extract_liver(self,liver=True):
#         """
#             get either the liver or tumor region from the mask
#             args:
#                 mask: mask of the liver and tumor
#                 liver: if True return the liver region, if False return the tumor region
#         """
#         mask = self._get_array(self.liver_seg)
#         mask = mask.astype(int)
#         print(mask.shape)
        
#         if liver:
#             mask[mask == 2] = 1
#         else:
#             mask[mask == 1] = 0
#             mask[mask == 2] = 1
#             mask = mask.astype(int)
            
#         # only keep the largest connected component
#         labeled = skimage.measure.label(mask, connectivity=2)

#         labeled[labeled != 1] = 0
      
#         mask = labeled
#         print(mask.sum())

#         return mask


        
#     def get_liver_bounding_box(self,liver_mask):
#         '''
#     Function to generate bounding box for liver, from a binary liver mask
#         args:
#             liver_mask: binary mask of the liver

#         returns:
#             bbox: bounding box of the liver (min_row, min_col, min_slice, max_row, max_col, max_slice = bbox)
#         '''

#         # get the image_probs
#         image_probs = skimage.measure.regionprops((liver_mask))

#         # get the bounding box of the liver

#         if len(image_probs) == 0:
#             print(f'[WARNING] no liver found')
#             self._recording_failing()
#             return None

#         ## find the adjacent box that contains the liver
#         for props in image_probs:
#             bbox = props.bbox
#             min_row, min_col, min_slice, max_row, max_col, max_slice = bbox 
#             print("this is range",min_row, min_col, min_slice, max_row, max_col, max_slice)
#         return [min_row, min_col, min_slice, max_row, max_col, max_slice]

#     def crop_scan(self,liver_bounding):
#         """
#         Crop the scan with the bounding box
#         args:
#             liver_bounding: the bounding box of the liver
#         """
#         liver_original = self._get_array(self.liver_orig)
#         # get the bounding box of the liver
#         min_row, min_col, min_slice, max_row, max_col, max_slice = list(map(self._check_range,liver_bounding))
#         print('this is after check range',min_row, min_col, min_slice, max_row, max_col, max_slice)
        
#         #crop the scan
#         cropped_scan = liver_original[min_row:max_row,min_col:max_col,min_slice:max_slice]
#         return cropped_scan
    
#     def store_cropped_data(self,cropped_data):

#     #     if not os.path.exists(self.out_path_box):
    #         os.mkdir(self.out_path_box)
    #         print('The path does not exist, create the path!')

    #     header = self.liver_orig.header
    #     affine = self.liver_orig.affine
    #     print("this is shape of cropped data",cropped_data.shape)
    #     cropped_image = nib.Nifti1Image(cropped_data, affine, header)


        
    #     nib.save(cropped_image, self.out_path_box + self.file_name)

    # @staticmethod   
    # def liver_detection(mask):
    #     if np.count_nonzero(mask) != 0:
    #         return True


    # def _check_range(self,range_num):
    #     return max(0,range_num)
    
    # def _get_array(self,image_file):
    #     return image_file.get_fdata()
    
    # def _recording_failing(self):
    #     with open(self.out_path_box + 'failing_box.txt','a') as f:
    #         f.write(self.file_name + '\n')
if __name__ == "__main__":

    # load the image
    #image_orig_path = '/data/scratch/r098986/CT_Phase/Data/Raw_Phase_data/'
    #image_seg_path = '/data/scratch/r098986/nnUnet_Seg/nnUNeT_test/Task_502_Phase_Data/'
    #cropped_out_path = '/data/scratch/r098986/CT_Phase/Data/Cropped_Data/'
    scans_info_path = '../Data/Mixed_HGP/Test/Test_scan_info.csv'
    image_path = '../Data/Mixed_HGP/Test_Data/'
    mask_path = '../Data/Mixed_HGP/Test_Mask_Data/'
    img_reader = ImageReader(ImageSet(image_path))
    mask_readr = ImageReader(ImageSet(mask_path))
    scans_info = FileLoad(scans_info_path).scans_df
    #load each image
    for i in range(scans_info.shape[0]):
        image = img_reader.load_image()
        mask = mask_readr.load_image()
        file_name = "CILM_" + scans_info.iloc[i]['Experiment'] + f"_{str(i)}" 
        image_name, mask_name = img_reader.load_image_name(), mask_readr.load_image_name()
        assert image.shape == mask.shape, "The shape of image and mask is not the same"
        liver_bbx = LiverBoundingBox(image,mask)
        liver_mask = liver_bbx.extract_liver()
        print(file_name,image_name,mask_name)
        break

    # image_orig_path = 'D:/Onedrive/bioinformatics_textbook/VU_Study/internship/Erasmusmc/Test_Data/Raw_Phase_data/'
    # image_seg_path = 'D:/Onedrive/bioinformatics_textbook/VU_Study/internship/Erasmusmc/Test_Data/Seg_Phase_data/'
    # cropped_out_path = '../../Test_Data/CT_Phase/Cropped_Data_Liver/'
    # #load the images
    # image_orig_load = ImageLoad(image_orig_path)
    # image_seg_load = ImageLoad(image_seg_path)
    # for i in range(image_orig_load.images_num):
    #     file_name = image_orig_load.images_names[i]
    #     image_orign,image_segg = image_orig_load.image_load(image_orig_load.image_path[i]),image_seg_load.image_load(image_seg_load.image_path[i])

    #     #finding bounding box
    #     liver_bbox = LiverBoundingBox(image_segg,image_orign,cropped_out_path,file_name)
    #     liver_mask = liver_bbox.extract_liver()
    #     if liver_bbox.liver_detection(liver_mask):
    #         liver_box_range = liver_bbox.get_liver_bounding_box(liver_mask)
    #         cropped_image = liver_bbox.crop_scan(liver_box_range)
    #         liver_bbox.store_cropped_data(cropped_image)
    #     else:
    #         liver_bbox._recording_failing()




    # # image_orig_nib,image_seg_nib = image_orig_load.images_nib,image_seg_load.images_nib
    # # #get the images
    # # for orig,seg in zip(image_orig_nib.items(),image_seg_nib.items()):
    # # #get the liver mask
    # # #create the liver bounding box
    # #     orig_name,seg_name = orig[0],seg[0]
    # #     orig_nib,seg_nib = orig[1],seg[1]
    # #     liver_bbox = LiverBoundingBox(original_image=orig_nib,liver_seg=seg_nib)
    # #     # get the bounding box of the liver
    # #     liver_mask = liver_bbox.extract_liver()
    # #     print(liver_mask.sum(),'hhhhhhhhhhh')
    # #     if liver_bbox.liver_detection(liver_mask):
    # #         liver_bounding = liver_bbox.get_liver_bounding_box(liver_mask)
    # #         cropped_image = liver_bbox.crop_scan(liver_bounding)
    # #         liver_bbox.store_cropped_data(cropped_image,'./Test_Data/Cropped_Data/',orig_name)
    # #         print(f"The file {orig_name} has liver! and the shape is {cropped_image.shape}")
    # #     else:
    # #         print(f"The file {orig_name} has no liver!")
    # #         with open('./Test_Data/Cropped_Data/No_Liver.txt','a') as f:
    # #             f.write(orig_name + '\n')

       
    #     # check if the mask is empty
        

        
    #     #print(cropped_image)
    # #mask_path = '/home/weiweiduan/Downloads/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task50_CILM/labelsTr/'
    # #image_name = 'CILM_1_0000.nii.gz'
    # #mask_name = 'CILM_1_0000.nii.gz'
    # # create the liver bounding box
    # #liver_bbox = LiverBoundingBox(mask_data,image_data)
    # # get the liver region
    # #liver_mask = liver_bbox.extract_liver()
    # # get the bounding box of the liver
    # #liver_bounding = liver_bbox.get_liver_bounding_box(liver_mask)
    # # crop the image
    # # cropped_image = liver_bbox.crop_scan(liver_bounding)
    # # # plot the image
    # # plt.imshow(cropped_image[:,:,0])
    # # plt.show()
    # # # save the image
    # # cropped_image_nifti = nib.Nifti1Image(cropped_image, image.affine, image.header