import cv2
import numpy as np
import os

palette = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]
classes = [0, 1, 2, 3, 4, 5, 6]

class_mapping = {tuple(color): cls for color, cls in zip(palette, classes)}

input_dir = "../data/land_cover/ann_dir/train"
output_dir = "../data/land_cover/ann_dir/grayscale"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for filename in os.listdir(input_dir):
    if filename.endswith('.png'):  
   
        img = cv2.imread(os.path.join(input_dir, filename))
        
        gray_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for color, cls in class_mapping.items():
            gray_img[np.all(img == color, axis=-1)] = cls
      
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, gray_img)
        print(f"Converted {filename} to grayscale.")

print("Conversion complete.")
