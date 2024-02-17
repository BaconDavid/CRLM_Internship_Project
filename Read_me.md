# Overall Structure

This Project code is divided into different parts, including main(run the model),Core(load model, dataloader, etc.),Pre-processing(as still testing different pre-processing methods, still not integrate)...

# How to use
## Core
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

## Pre-processing
This module includes the steps to pre-processing data. Not implemented yet and will add them in the future.
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



# Experiments 
## All tumors
## Largest tumor
1. 20 extra slices
## Per tumor
1. only per tumor 
2. per tumor bounding box