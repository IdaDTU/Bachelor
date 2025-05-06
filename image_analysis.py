import numpy as np
import matplotlib.pyplot as plt

def grayscale_to_colored(img_gray, min_val, max_val, cmap='viridis'):
    """
    De-normalizes a grayscale image and applies a colormap.

    Parameters:
        img_gray : 2D numpy array (grayscale image, values normalized between 0 and 1 or unknown)
        min_val  : original minimum value
        max_val  : original maximum value
        cmap     : string name of a matplotlib colormap or a colormap object

    Returns:
        RGB image as a (H, W, 3) numpy array with dtype uint8
    """
    # De-normalize
    img_real = img_gray * (max_val - min_val) + min_val

    # Normalize to 0–1 for colormap application
    img_normalized = (img_real - min_val) / (max_val - min_val)
    img_normalized = np.clip(img_normalized, 0, 1)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    img_colored = colormap(img_normalized)  # returns RGBA

    # Drop alpha channel and convert to uint8
    img_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)

    return img_rgb



def subtract_images(img1, img2):
    """
    Subtract img2 from img1. Both must be NumPy arrays of the same shape.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape to subtract.")

    result = img1.astype(np.float32) - img2.astype(np.float32)
    return result
