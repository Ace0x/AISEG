import cv2
import numpy as np

# Path to the image file
image_path = "../data/land_cover/ann_dir/val/119_mask.png"
output_file = "output_matrix.txt"

# Read the image
img = cv2.imread(image_path)
print(np.unique(img))



