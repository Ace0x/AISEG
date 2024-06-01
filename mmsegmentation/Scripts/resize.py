import os
import cv2
from pathlib import Path

# Path to the folder containing the images
input_folder = '../data/land_cover/img_dir/train'
output_folder = '../data/land_cover/img_dir/retrain'
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Resize the image to 256x256
        resized_image = cv2.resize(image, (256, 256))

        # Save the resulting image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)

print("Processing complete. Resized images saved to", output_folder)
