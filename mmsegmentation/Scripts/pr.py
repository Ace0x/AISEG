import cv2
import numpy as np

# Path to the image file
image_path = "../data/land_cover/ann_dir/train/119_mask.png"
output_file = "output_matrix.txt"

# Read the image
img = cv2.imread(image_path)

# Ensure the image is not None
if img is not None:
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image matrix to a file
    with open(output_file, 'w') as f:
        for row in gray_img:
            f.write(' '.join(map(str, row)) + '\n')
    
    print("Grayscale Image Matrix saved to:", output_file)
else:
    print("Error: Unable to read the image.")

print(np.unique(gray_img))