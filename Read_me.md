# Overall Structure

This Project code is divided into different parts, including main(run the model),Core(load model, dataloader, etc.),Pre-processing(going to make it a pipline)...

# How to use

## Now
We run the main file to run the model, here are all the pars:

1. data_path
2. label_path
3. epochs 
4. mode -- train or test
5. result_save_path -- where to save result
6. device -- cpu or cuda
7. Debug --Use subset to run

## How to run once change the model, dataset,num_class...etc?

Since we run single epoch in a function, we only need to change these in main file:

1. **NUM_CLASS** 
2. **build_model**


the rest of other you can keep the same. However, you can change build model, dataloader,...etc, if you want to add more model or functions.

