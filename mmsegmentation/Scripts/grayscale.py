import os
import cv2
import numpy as np
from pathlib import Path

# Define the color classes with their exact values
color_classes = {
    'unknown': {'color': [0, 0, 0], 'greyscale': 0},
    'urban_land': {'color': [0, 255, 255], 'greyscale': 1},
    'agriculture_land': {'color': [255, 255, 0], 'greyscale': 2},
    'rangeland': {'color': [255, 0, 255], 'greyscale': 3},
    'forest_land': {'color': [0, 255, 0], 'greyscale': 4},
    'water': {'color': [0, 0, 255], 'greyscale': 5},
    'barren_land': {'color': [255, 255, 255], 'greyscale': 6}
}

# Function to map colors to greyscale
def map_color_to_greyscale(image, color_classes):
    height, width, _ = image.shape
    greyscale_image = np.zeros((height, width), dtype=np.uint8)

    for color_class in color_classes.values():
        color = np.array(color_class['color'], dtype=np.uint8)
        grey_value = color_class['greyscale']

        # Create a mask where the pixel matches the class color
        mask = np.all(image == color, axis=-1)
        greyscale_image[mask] = grey_value

    return greyscale_image

# Path to the folder containing the masks
input_folder = '../data/land_cover/ann_dir/greytrain'
output_folder = '../data/land_cover/ann_dir/train'
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Map colors to greyscale
        greyscale_image = map_color_to_greyscale(image, color_classes)

        # Save the resulting image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, greyscale_image)

print("Processing complete. Greyscale masks saved to", output_folder)
