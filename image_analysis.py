import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from data_visualization import dtu_grey  # assuming you're not using dtu_coolwarm_cmap for plotting


from skimage.color import rgb2gray

def compute_tb(image_path, tb_min, tb_max):
    """
    Compute brightness temperature from RGB image.

    Parameters:
    image_path (str): Path to the RGB image file
    tb_min (float): Minimum brightness temperature (K)
    tb_max (float): Maximum brightness temperature (K)

    Returns:
    tb_map (np.ndarray): Brightness temperature map
    rgb_array (np.ndarray): Original RGB array normalized to [0,1]
    """
    rgb_img = Image.open(image_path).convert("RGB")
    rgb_array = np.array(rgb_img).astype(np.float32) / 255.0
    gray_intensity = rgb2gray(rgb_array)
    tb_map = tb_min + gray_intensity * (tb_max - tb_min)
    return tb_map

