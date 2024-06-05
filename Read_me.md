# Overall Structure

This Project code is divided into different parts, including main(run the model),Core(load model, dataloader, etc.),Pre-processing(as still testing different pre-processing methods, still not integrate)...

# Core Part
This part includes the main model, dataloader, and configuration file.
## Dataset

### Data download from XNAT and nnUnet
**Pay attention**: When extracting downloaded data, scan ids are used to label and rank each sample. For some samples, their scan id is not a single number, but only few of them are in such conditions. Therefore, we need to check the scan id inside each experiment and mannually assign a unique number to each sample. eg: A patient has 3 scans, and two of them are named as "1" and "2", while the third one is 2-CT10, so we need to manually assign "3" to the third scan by changing the raw data file name. However, I recommend add a new column :scan_id instead of replacing their original scan name

**Important**: Make the scans number correctly is very important when pre-processing the data, and make sure scans id rank correctly from small to large.
### Config
This is the folder that contains model configuration file. When running the model, only need to load the configuration file and mode. In configuration file, here below are some essential part you need to assign:
1. Model name
2. Save dir
3. Data csv file that must include: data path and label name.
4. fold

Then the corresponding results will automatically be saved in the save folder that you assign.
### Dataset
This folder includes code to build dataset and dataloader. You can design the parameters yourself. It is inherited from class ImageDataset and Dataloder from monai. you can operate what you want in the inherited class(eg:print info, modify img or label in dataset...)

### Utils
This module includes some basic tools, like calculating metrics, building models, and some modifed models. In Utility script it has some tools functions such as path check.


## Evaluation
This module contains scripts for results analysis. In the future might add some visualization methods.

## Jupyter Test
This contains script to test code by interactive jupyter notebook. They are a little bit in mess but are divided into different test parts to make it easier to replicate for different aims(such as for data processing)
## Source code
contains some models of their corresponding source code.

If you want to run the model simply and directly , Here below are the steps:
1. First make sure you have a data info csv file, and have **label**, **img path** column.(Will put an example file here)
2. Then set a proper configureation file, which includes essential parts as mentioned above.
3. Run "Python main.py --config_file xxx.yaml --mode "train/test"


the rest of other you can keep the same. However, you can change build model, dataloader,...etc, if you want to add more model or functions.

# Pre-processing

As we all know, the most annoying part is pre-processing. As there are all tumors, per tumor, largest tumor, we need to preprocess them separately. 

First: Download the data from XNAT, I recommend first you store the data info in a csv file, and then you can use it to download the data. Besides, you can also download from xnat directly, you can only download earliest scan or the whole scans. **Remeber, there are some scan IDs are not numbers**, I recommend to manually assign a unique number to each sample, better from small to large, and add a new column :scan_id instead of replacing their original scan name.

Second: move all mask files and raw data files to a folder, name format: "CILM_{Exp_name}_{scan_id}.nii.gz". For example, one 
Patient has 2 scans, then the name of mask and raw image files should be:

CILM_CT_10033_0.nii.gz
CILM_CT_10033_1.nii.gz

Third: Extract liver and tumor
Before cropping liver and tumor, prepare for the correct csv file. The csv file should include:
1. Sample Column: eg:CILM_CT_10033_0
2. tumor_id: eg: 0,1,2...
3. crop range: min_z,min_y,min_x,max_z,max_y,max_x

# Experiments 
## All tumors
## Largest tumor
1. 20 extra slices
## Per tumor
1. only per tumor 
2. per tumor bounding box


## Evaluation
Keep every output as a single output from batch size one, and finally calculate them by stacking






