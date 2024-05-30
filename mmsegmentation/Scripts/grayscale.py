import cv2
import numpy as np
import os

# Define the color palette and class mappings
palette = [
    [0, 255, 255],   # urban_land
    [255, 255, 0],   # agriculture_land
    [255, 0, 255],   # rangeland
    [0, 255, 0],     # forest_land
    [0, 0, 255],     # water
    [255, 255, 255], # barren_land
    [0, 0, 0]        # unknown
]

classes = [0, 1, 2, 3, 4, 5, 6]

# Create a mapping from RGB color to class index
class_mapping = {tuple(color): cls for color, cls in zip(palette, classes)}

input_dir = "../data/land_cover/ann_dir/train"
output_dir = "../data/land_cover/ann_dir/grayscale"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  
        # Read the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read {img_path}. Skipping...")
            continue

        # Initialize the grayscale image with the default class for 'unknown'
        gray_img = np.full((img.shape[0], img.shape[1]), class_mapping[(0, 0, 0)], dtype=np.uint8)

        # Convert the image to grayscale based on the color mapping
        for color, cls in class_mapping.items():
            mask = np.all(img == np.array(color), axis=-1)
            gray_img[mask] = cls
        
        # Save the grayscale image
        output_path = os.path.join(output_dir, filename)
        if cv2.imwrite(output_path, gray_img):
            print(f"Converted {filename} to grayscale.")
        else:
            print(f"Failed to write {output_path}.")
        
print("Conversion complete.")

