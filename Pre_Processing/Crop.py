###### Liver Bounding Box ######
# This script is used to generate liver bounding box for each patient
# The bounding box is used to crop the liver region from the original image

from abc import ABC, abstractmethod
import os
from random import choices, sample
from cv2 import sort
import nibabel as nib
import numpy as np
import skimage 
import matplotlib.pyplot as plt
from sympy import Union
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
from typing import Union


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
        self.image_names = sorted((os.listdir(self.image_path)))
        self.image_num = len(self.image_names)
        self.image_full_paths = [os.path.join(self.image_path, name) for name in self.image_names]
        print("Total number of images:", self.image_num)

    def get_image_list(self):
        return self.image_names
    
    def get_image_num(self):
        return self.image_num
    
    def get_image_full_path(self, index):
        if index < 0 or index > self.image_num:
            raise IndexError("Index out of range")
        
        return self.image_full_paths[index]



class ImageReader:
    def __init__(self,image_set):
        self.image_set = image_set
        self.current_index = 0
        self.current_name_index = 0

    def load_image(self,reader='sitk'):
        if self.current_index > self.image_set.get_image_num():
            raise StopIteration("No more images to read")
        image_path = self.image_set.get_image_full_path(self.current_index)

        if reader == 'nib':
            image = nib.load(image_path)
            image_array = nib.load(image_path).get_fdata()
        elif reader == 'sitk':
            #get image array
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
        else:
            raise ValueError("The reader should be either nib or sitk!")
        self.current_index += 1
        return image, image_array

    def load_image_name(self):
        if self.current_index > self.image_set.get_image_num():
            raise StopIteration("No more images to read")
        image_name = self.image_set.get_image_list()[self.current_name_index]
        self.current_name_index += 1
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
        liver_mask = self.extract_liver()
        image_probs = skimage.measure.regionprops(liver_mask)

        # get the bounding box of the liver

        if len(image_probs) == 0:
            print(f'[WARNING] {self.file_name} no liver found')

        ## find the adjacent box that contains the liver
        ones_indices = np.argwhere(liver_mask == 1)
        # find the bounidng box in each dimention
        if ones_indices.size > 0:
            min_z, min_y, min_x = ones_indices.min(axis=0)
            max_z, max_y, max_x = ones_indices.max(axis=0)

        return [(min_z, min_y, min_x, max_z, max_y+1, max_x+1)] # add 1 to include the last pixel


class TumorBoundingBox(ABC):
    def __init__(self,liver_img,liver_mask:np.ndarray = None) -> None:
        self.liver_img = liver_img
        self.liver_mask = liver_mask
    
    @abstractmethod
    def extract_tumor(self):
        pass

    @abstractmethod
    def get_tumor_bounding_box(self) -> list:
        """
        return a list containing min_row, min_col, min_slice, max_row, max_col, max_slice of the tumor
        """
        pass


class LargestTumorBoundingBox(TumorBoundingBox):
    def __init__(self,liver_img,liver_mask:np.ndarray = None) -> None:
        super().__init__(liver_img,liver_mask)
    
    def extract_tumor(self):
        mask = self.liver_mask
        mask = mask.astype(int)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        #extract largest connected component
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
    
    def get_tumor_bounding_box(self) -> list:
        """
        return a list containing min_row, min_col, min_slice, max_row, max_col, max_slice of the tumor
        """
        tumor_mask = self.extract_tumor()
        per_tumor_bounding = []
        tumor_size = tumor_mask.sum()
        ones_indices = np.argwhere(tumor_mask == 1)
        # find the bounidng box in each dimention
        if ones_indices.size > 0:
            min_z, min_y, min_x = ones_indices.min(axis=0)
            max_z, max_y, max_x = ones_indices.max(axis=0)

        #calculate tumor size
        #tumor_size = tumor_mask.sum()
        per_tumor_bounding.append((min_z, min_y, min_x, max_z, max_y, max_x, tumor_size))
        
        return per_tumor_bounding

class PerTumorBoundingBox(TumorBoundingBox):
    def __init__(self,liver_img,liver_mask:np.ndarray = None) -> None:
        super().__init__(liver_img,liver_mask)
    
    def extract_tumor(self):
        mask = self.liver_mask
        mask = mask.astype(int)
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        per_tumor_list = []
        labeled,num_features = skimage.measure.label(mask, connectivity=2,return_num=True)
        for label in range(1, num_features + 1):
            component_mask = np.where(labeled == label, 1, 0)
            per_tumor_list.append(component_mask)
        
        #stack each tumor mask
        per_tumor_mask = np.stack(per_tumor_list)
        
        return per_tumor_mask
        
    def get_tumor_bounding_box(self) -> list:
        """
        return a list containing min_row, min_col, min_slice, max_row, max_col, max_slice of each tumor
        """
        per_tumor_mask = self.extract_tumor()
        per_tumor_bounding = []
        for i in range(per_tumor_mask.shape[0]):  # how many tumors
            # get the mask of each tumor
            tumor_mask_i = per_tumor_mask[i, :, :,:]
            tumor_size_i = tumor_mask_i.sum()

            ones_indices = np.argwhere(tumor_mask_i == 1)
            # find the bounidng box in each dimention
            if ones_indices.size > 0:
                min_z, min_y, min_x = ones_indices.min(axis=0)
                max_z, max_y, max_x = ones_indices.max(axis=0)

            #calculate tumor size
            #tumor_size = tumor_mask_i.sum()
            
            per_tumor_bounding.append((min_z, min_y, min_x, max_z, max_y, max_x, tumor_size_i))

        return per_tumor_bounding

class TumorBoundingBoxFactory:
    @staticmethod
    def create_tumor_bounding_box(tumor_type='largest',liver_img=None,liver_mask=None):
        assert tumor_type in ['largest','per_tumor'], "Tumor type should be either 'largest' or 'per_tumor'"
        if tumor_type == 'largest':
            return LargestTumorBoundingBox(liver_img,liver_mask)
        elif tumor_type == 'per_tumor':
            return PerTumorBoundingBox(liver_img,liver_mask)

def generate_bounding_df(data,type='liver'):
    """
    Convert a dictionary of samples to a DataFrame.
    
    Parameters:
    data (dict): A dictionary where keys are sample names and values are lists of tuples. 
                 Each tuple contains three elements representing the values in three columns.
                 
    Returns:
    pd.DataFrame: A DataFrame with four columns: 'col1', 'col2', 'col3', and 'sample'.
                  Each row corresponds to one tuple from the input dictionary, 
                  with an additional column 'sample' indicating the sample name.
    """
    # Initialize an empty list to store all DataFrames
    dfs = []

    # Iterate over each key-value pair in the dictionary
    for key, values in data.items():
        # Create a DataFrame for each list of tuples
        if type == 'liver':
            df = pd.DataFrame(values, columns=['min_z','min_y','min_x','max_z','max_y','max_x'])
        elif type == 'tumor':
            df = pd.DataFrame(values, columns=['min_z','min_y','min_x','max_z','max_y','max_x','tumor_size'])
            df['tumor_id'] = range(len(values)) # give each tumor a unique id in a sample

        # Add a new column to indicate the sample name
        df['sample'] = key
        # Append the DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames into one
    result = pd.concat(dfs, ignore_index=True, axis=0)
    
    # Return the resulting DataFrame
    return result
def extend_margin_slice(crop_df,image_array,slice_margin: Union[int ,float] = 5,threshold=None):

    """
    args:
        crop_df: the dataframe containing the liver bounding box info
    """
    #initialize new columns with nan
    crop_df['extension_min_z'] = np.nan
    crop_df['extension_max_z'] = np.nan

    max_z_dim = image_array.shape[0] #upper bound is the last slice

    for i in range(crop_df.shape[0]):
        min_z, max_z = crop_df.loc[i,['min_z','max_z']].values.astype(int)
        if isinstance(slice_margin,int):
            extension_slice = slice_margin
        elif isinstance(slice_margin,float):
            extension_slice = int(max_z - min_z) * slice_margin
        
        new_min_z, new_max_z = max(0,min_z-extension_slice), min(max_z_dim,max_z+extension_slice) #lower bound is 0 and upper bound is the last slice
        print(new_min_z,new_max_z,'new_min_z,new_max_z')
        #if we set a threshold for slices limitation

        if threshold is not None and (max_z - min_z) >= threshold:
            new_min_z, new_max_z = min_z, max_z # keep the same if the slices are under the threshold.
            
        crop_df.loc[i, 'extension_min_z'] = new_min_z
        crop_df.loc[i, 'extension_max_z'] = new_max_z        
        print(crop_df.loc[i,['min_z','max_z','extension_min_z','extension_max_z']])
    return crop_df

def crop_liver_tumor(crop_df,reader = 'sitk',tumor_type='largest',img_path=None,mask_path=None,output_img_path=None,output_path=None):
    for i in range(crop_df.shape[0]):
        sample = crop_df.loc[i,'sample']
        tumor_id = crop_df.loc[i,'tumor_id']
        liver_img_path = os.path.join(image_path,sample+'.nii.gz')
        liver_mask_path = os.path.join(mask_path,sample+'.nii.gz')
        min_z, min_y, min_x, max_z, max_y, max_x = crop_df.loc[i,['min_z','min_y','min_x','max_z','max_y','max_x']].values.astype(int)
        if reader == 'sitk':
            mask, mask_array = sitk.ReadImage(liver_img_path), sitk.GetArrayFromImage(sitk.ReadImage(liver_mask_path))
            image, image_array = sitk.ReadImage(liver_img_path), sitk.GetArrayFromImage(sitk.ReadImage(liver_img_path))
            extracted_img_array = image_array[min_z:max_z,min_y:max_y,min_x:max_x]
            print(extracted_img_array.shape,'shitttt')
            #write new cropped image
            extracted_img = sitk.GetImageFromArray(extracted_img_array)
            extracted_img.SetSpacing(image.GetSpacing())
            extracted_img.SetDirection(image.GetDirection())
            extracted_img.SetOrigin(image.GetOrigin()) 
            #same for mask          
            image_extracted_name = sample + '_' + str(tumor_id) + '.nii.gz'
            output_img_path = os.path.join(output_path,image_extracted_name)
            print(output_img_path,'output_img_path')
            sitk.WriteImage(extracted_img, output_img_path)#image_path
        
        elif reader == 'nib':
            mask, mask_array = nib.load(liver_mask_path), nib.load(liver_mask_path).get_fdata()
            image, image_array = nib.load(liver_img_path), nib.load(liver_img_path).get_fdata()
            pass #if you want to use nibabel reader, you need to install nibabel







if __name__ == "__main__":
    scans_info_path = '../../Data/Mixed_HGP/True_Label/scans_used_all_info_with_tumor.csv'
    image_path = '../../Data/Mixed_HGP/Mixed_HGP_07071/'
    mask_path = '../../Data/Mixed_HGP/Mixed_HGP_mask_07071/'
    img_reader = ImageReader(ImageSet(image_path))
    mask_readr = ImageReader(ImageSet(mask_path))
    scans_info = FileLoad(scans_info_path).scans_df
    #load each image
    liver_bbx_dict = {}
    tumor_bbx_dict = {}


    img, img_array = sitk.ReadImage('../../Data/Mixed_HGP/Mixed_HGP_mask_07071/CILM_CT_27232_0.nii.gz'), sitk.GetArrayFromImage(sitk.ReadImage('../../Data/Mixed_HGP/Mixed_HGP_mask_07071/CILM_CT_27232_0.nii.gz'))

    mask, mask_array = sitk.ReadImage('../../Data/Mixed_HGP/Mixed_HGP_mask_07071/CILM_CT_27232_0.nii.gz'), sitk.GetArrayFromImage(sitk.ReadImage('../../Data/Mixed_HGP/Mixed_HGP_mask_07071/CILM_CT_27232_0.nii.gz'))


    liver_bbx = LiverBoundingBox(img_array,mask_array)
    tumor_bbx = TumorBoundingBoxFactory().create_tumor_bounding_box("largest",img_array,mask_array)
    liver_bounding = liver_bbx.get_liver_bounding_box()
    tumor_bounding_size = tumor_bbx.get_tumor_bounding_box()

    # #store info
    liver_bbx_dict['CILM_CT_27354_0'] = liver_bounding
    tumor_bbx_dict['CILM_CT_27354_0'] = tumor_bounding_size
    liver_bbx_df = generate_bounding_df(liver_bbx_dict,type='liver')
    tumor_bbx_df = generate_bounding_df(tumor_bbx_dict, type='tumor')
    tumor_bbx_extend_df = extend_margin_slice(tumor_bbx_df, img_array, slice_margin=5, threshold=17)
    liver_bbx_extend_df = extend_margin_slice(liver_bbx_df, img_array, slice_margin=5, threshold=17)
    
    print(tumor_bbx_extend_df,tumor_bbx_dict)
        
    for i in range(scans_info.shape[0]):

        #create dic to store the bounding box
        image, image_array = img_reader.load_image()
        mask, mask_array = mask_readr.load_image()
        image_name, mask_name = img_reader.load_image_name(), mask_readr.load_image_name()
        print(image_name,mask_name,'666')
        file_name = "CILM_" + scans_info.iloc[i]['Experiment'] + "_" + str(scans_info.iloc[i]['Scan_Id']) #remember to have such two columns! 
        
        print(file_name,"processing!")
        assert image_array.shape == mask_array.shape, "The shape of image and mask is not the same"

        liver_bbx = LiverBoundingBox(image_array,mask_array)
        tumor_bbx = TumorBoundingBoxFactory().create_tumor_bounding_box("largest",image_array,mask_array)
        liver_bounding = liver_bbx.get_liver_bounding_box()
        tumor_bounding_size = tumor_bbx.get_tumor_bounding_box()
        
        #store info
        liver_bbx_dict[file_name] = liver_bounding
        tumor_bbx_dict[file_name] = tumor_bounding_size
        print(tumor_bbx_dict)
        

        

    
    #crop
    liver_bbx_df = generate_bounding_df(liver_bbx_dict,type='liver')
    tumor_bbx_df = generate_bounding_df(tumor_bbx_dict, type='tumor')
    #extend margin slice
    tumor_bbx_extend_df = extend_margin_slice(tumor_bbx_df, image_array, slice_margin=5, threshold=17)
    liver_bbx_extend_df = extend_margin_slice(liver_bbx_df, image_array, slice_margin=5, threshold=17)
    #crop_liver_tumor(liver_bbx_extend_df,tumor_type='largest',img_path=image_path,mask_path=mask_path,output_path= "../Data/Test/")
    print(liver_bbx_df,tumor_bbx_df)
    tumor_bbx_extend_df.to_csv('../../Data/Test/tumor_lg_mix_df.csv')

    liver_bbx_extend_df.to_csv('../../Data/Test/liver_lg_mix_df.csv')
    


    