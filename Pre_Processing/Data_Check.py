"""
To check the data properties
"""
from Crop import ImageLoad
import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(data):
    """
    Plot the distribution of the data's height,width and depth
    data:CT scan data after resampling|arrary with shape (3,sample_nums)
    """

    fig,ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].hist(data[0,:],bins=50)
    ax[0].set_title('Height')
    ax[1].hist(data[1,:],bins=50)
    ax[1].set_title('Width')
    ax[2].hist(data[2,:],bins=50)
    ax[2].set_title('Depth')
    plt.show()
   

if __name__ == '__main__':
    image_path = './Test_Data/Resampled_Phase_Data/'
    image_load = ImageLoad(image_path)
    #create a array(3X sample_nums)
    images_num = image_load.images_num
    images_data = np.ndarray((3,images_num),dtype=object)
    for i in range(image_load.images_num):
        image_path = image_load.image_path[i]
        image_float = image_load.image_load(image_path).get_fdata()
        images_data[0,i],images_data[1,i],images_data[2,i] = image_float.shape
    
    #plot the distribution
    plot_distribution(images_data)