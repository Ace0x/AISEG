import os
import numpy as np
from PIL import Image

def compute_mean_std(image_folder):
    # Initialize sums and counters for mean calculations
    sum_r = 0
    sum_g = 0
    sum_b = 0
    total_pixels = 0

    # First pass: calculate mean
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img = Image.open(os.path.join(image_folder, filename)).convert('RGB')
            img_array = np.array(img)
            
            red = img_array[:, :, 0]
            green = img_array[:, :, 1]
            blue = img_array[:, :, 2]
            
            sum_r += np.sum(red)
            sum_g += np.sum(green)
            sum_b += np.sum(blue)
            
            total_pixels += red.size
    
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels
    
    # Initialize sums for standard deviation calculations
    sum_r_sq = 0
    sum_g_sq = 0
    sum_b_sq = 0

    # Second pass: calculate standard deviation
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img = Image.open(os.path.join(image_folder, filename)).convert('RGB')
            img_array = np.array(img)
            
            red = img_array[:, :, 0]
            green = img_array[:, :, 1]
            blue = img_array[:, :, 2]
            
            sum_r_sq += np.sum((red - mean_r) ** 2)
            sum_g_sq += np.sum((green - mean_g) ** 2)
            sum_b_sq += np.sum((blue - mean_b) ** 2)
    
    std_r = np.sqrt(sum_r_sq / total_pixels)
    std_g = np.sqrt(sum_g_sq / total_pixels)
    std_b = np.sqrt(sum_b_sq / total_pixels)

    mean = [mean_r, mean_g, mean_b]
    std = [std_r, std_g, std_b]

    return mean, std

# Path to the folder containing images
image_folder = '../data/land_cover/img_dir/train'

mean, std = compute_mean_std(image_folder)

print(f'Final Mean (RGB): {mean}')
print(f'Final Standard Deviation (RGB): {std}')
