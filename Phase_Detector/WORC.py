# %%
### imports ###
from WORC import BasicWORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob
import numpy as np

# Set paths to the data

# current working directory
#path = os.getcwd()
image_file_name = 'image.nii.gz'
segmentation_file_name = 'mask.nii.gz'


script_path = '/trinity/home/r098986/CRLM_Intership_Project_Run/CRLM_Internship_Project/Phase_Detector/'


data_path = '/data/scratch/r098986/CT_Phase/Data/Full_Image_Liver'

# File in which the labels (i.e. outcome you want to predict) is stated
# Again, change this accordingly if you use your own data.
label_file = '/data/scratch/r098986/CT_Phase/Data/True_Label/' + 'Phase_PVP.csv'


# Name of the label you want to predict
# Classification: predict a binary (0 or 1) label
label_name = ['Phase']

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = False

# Give your experiment a name
experiment_name = '2_Phase_Detection'

print(experiment_name)

# Instead of the default tempdir, let's but the temporary output in a subfolder
# in the same folder as this script
tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)

# %%
# make dicts in which the images and segmentations are stored
# following format: 
# images1 = {'patient1_0': '/data/Patient1/image_MR.nii.gz', 'patient1_1': '/data/Patient1/image_MR.nii.gz'}
# segmentations1 = {'patient1_0': '/data/Patient1/seg_tumor1_MR.nii.gz', 'patient1_1': '/data/Patient1/seg_tumor2_MR.nii.gz'}


#imagedatadir = data_path + 'task_507_CRLM_MIX_PVP_best/'
imagedatadir = data_path
images = {}
segmentations = {}
label_df = pd.read_csv(label_file)['Patient'].values

for folder_name in os.listdir(imagedatadir):
    
    # check if img is in label file, if not do not include image
    if folder_name in label_df:
      pass
    else:
      continue
      
    for file_name in os.listdir(imagedatadir + folder_name):
        
        if file_name.startswith('mask'):   
            
            images[folder_name + file_name[4:8]] = imagedatadir + folder_name  +'/image.nii.gz'
            segmentations[folder_name + file_name[4:8]] =  imagedatadir + folder_name  +'/mask.nii.gz'
       

print("images",images)
print("segmentations",segmentations)
# %%
experiment = BasicWORC(experiment_name)

# Set the input data according to the variables we defined earlier
experiment.images_train.append(images)
experiment.segmentations_train.append(segmentations)

experiment.labels_from_this_file(label_file)
experiment.predict_labels(label_name)

experiment.set_image_types(['CT'])

experiment.binary_classification(coarse=coarse)

# give more memory & solve problem of dimensions mask and images
overrides = {
          'General': {
              'AssumeSameImageAndMaskMetadata': 'True',
              },
           'HyperOptimization': {
               'memory': '9G',
              }, 
           'Imputation': {
              'skipallNaN': 'True',
              },    
            }

experiment.add_config_overrides(overrides)


experiment.add_evaluation()
# Run the experiment!
experiment.execute()

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

print(f"Your output is stored in {experiment_folder}.")