import os
import cv2
import numpy as np
from pathlib import Path

# Define the color classes with their lower and upper boundaries
color_classes = {
    'urban_land': {'lower': [0, 245, 245], 'upper': [10, 255, 255], 'greyscale': 0},
    'agriculture_land': {'lower': [245, 245, 0], 'upper': [255, 255, 10], 'greyscale': 1},
    'rangeland': {'lower': [245, 0, 245], 'upper': [255, 10, 255], 'greyscale': 2},
    'forest_land': {'lower': [0, 245, 0], 'upper': [10, 255, 10], 'greyscale': 3},
    'water': {'lower': [0, 0, 245], 'upper': [10, 10, 255], 'greyscale': 4},
    'barren_land': {'lower': [245, 245, 245], 'upper': [255, 255, 255], 'greyscale': 5},
    'unknown': {'lower': [0, 0, 0], 'upper': [10, 10, 10], 'greyscale': 6}
}

# Function to map colors to greyscale
def map_color_to_greyscale(image, color_classes):
    height, width, _ = image.shape
    greyscale_image = np.zeros((height, width), dtype=np.uint8)

    for color_class in color_classes.values():
        lower = np.array(color_class['lower'], dtype=np.uint8)
        upper = np.array(color_class['upper'], dtype=np.uint8)
        grey_value = color_class['greyscale']

        # Create a mask where the pixel falls within the color boundaries
        mask = cv2.inRange(image, lower, upper)
        greyscale_image[mask > 0] = grey_value

    return greyscale_image

# Paths to the folders
mask_input_folder = '../data/land_cover/train/original'
sat_input_folder = '../data/land_cover/train/original'
mask_output_folder = '../data/land_cover/train/mask'
sat_output_folder = '../data/land_cover/train/sat'
Path(mask_output_folder).mkdir(parents=True, exist_ok=True)
Path(sat_output_folder).mkdir(parents=True, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(mask_input_folder):
    if filename.endswith('_mask.png'):
        # Read the mask
        mask_path = os.path.join(mask_input_folder, filename)
        mask = cv2.imread(mask_path)

        # Convert from BGR to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Resize the mask to 256x256
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Map colors to greyscale
        greyscale_mask = map_color_to_greyscale(mask, color_classes)

        # Read the corresponding satellite image
        base_filename = filename.replace('_mask.png', '')
        sat_filename = f"{base_filename}_sat.jpg"
        sat_path = os.path.join(sat_input_folder, sat_filename)
        if not os.path.exists(sat_path):
            print(f"Satellite image for {filename} not found, skipping.")
            continue
        sat_image = cv2.imread(sat_path)

        # Resize the satellite image to 256x256
        sat_image = cv2.resize(sat_image, (256, 256))

        # Save the resulting mask and satellite image
        mask_output_path = os.path.join(mask_output_folder, filename)
        sat_output_path = os.path.join(sat_output_folder, sat_filename)
        cv2.imwrite(mask_output_path, greyscale_mask)
        cv2.imwrite(sat_output_path, sat_image)

        # Check if the mask contains the 'unknown' class
        if 6 in greyscale_mask:
            # Save additional copies for oversampling
            for i in range(3):  # Adjust the number of copies as needed
                mask_oversample_output_path = os.path.join(mask_output_folder, f"{base_filename}_mask_copy{i}.png")
                sat_oversample_output_path = os.path.join(sat_output_folder, f"{base_filename}_sat_copy{i}.jpg")
                cv2.imwrite(mask_oversample_output_path, greyscale_mask)
                cv2.imwrite(sat_oversample_output_path, sat_image)

print("Processing complete. Greyscale masks and satellite images saved to their respective folders.")
