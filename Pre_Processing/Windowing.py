from matplotlib import pyplot as plt
import SimpleITK as sitk
import numpy as np


def apply_window_to_volume(batched_volumes, window_center, window_width):
    """
    Apply windowing to a batch of 3D volumes.
    :param batched_volumes: The input batch of 3D volumes.
    :param window_center: The center of the window (window level).
    :param window_width: The width of the window.
    :return: Windowed batch of 3D volumes.
    """
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    windowed_batched_volumes = np.clip(batched_volumes, lower_bound, upper_bound)
    return windowed_batched_volumes


image = sitk.ReadImage("../../Data/CT_Phase/Full_Image_Liver_07075/CILM_CT_101040_0000.nii.gz")
image = sitk.GetArrayFromImage(image)

windowed_image = apply_window_to_volume(image, 50, 400)

plt.imshow(windowed_image[80,:,:], cmap='gray')