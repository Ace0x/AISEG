import os
import cv2
import numpy as np
from pathlib import Path

# Define the color classes with their lower and upper bounds
color_classes = {
    'urban_land': {'range': [(0, 225, 225), (30, 255, 255)], 'greyscale': 0},
    'agriculture_land': {'range': [(225, 225, 0), (255, 255, 30)], 'greyscale': 1},
    'rangeland': {'range': [(225, 0, 225), (255, 30, 255)], 'greyscale': 2},
    'forest_land': {'range': [(0, 225, 0), (30, 255, 30)], 'greyscale': 3},
    'water': {'range': [(0, 0, 225), (30, 30, 255)], 'greyscale': 4},
    'barren_land': {'range': [(225, 225, 225), (255, 255, 255)], 'greyscale': 5},
    'unknown': {'range': [(0, 0, 0), (30, 30, 30)], 'greyscale': 6}
}

# Function to map colors to greyscale
def map_color_to_greyscale(image, color_classes):
    height, width, _ = image.shape
    greyscale_image = np.zeros((height, width), dtype=np.uint8)

    for color_class in color_classes.values():
        lower_bound = np.array(color_class['range'][0], dtype=np.uint8)
        upper_bound = np.array(color_class['range'][1], dtype=np.uint8)
        grey_value = color_class['greyscale']

        mask = cv2.inRange(image, lower_bound, upper_bound)
        greyscale_image[mask > 0] = grey_value

    return greyscale_image

# Path to the folder containing the masks
input_folder = '../data/land_cover/ann_dir/grayt'
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
