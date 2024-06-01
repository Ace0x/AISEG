import os
import cv2
import numpy as np
from pathlib import Path

# Define the color classes with their exact values
color_classes = {
    'urban_land': {'color': [0, 255, 255], 'greyscale': 0},
    'agriculture_land': {'color': [255, 255, 0], 'greyscale': 1},
    'rangeland': {'color': [255, 0, 255], 'greyscale': 2},
    'forest_land': {'color': [0, 255, 0], 'greyscale': 3},
    'water': {'color': [0, 0, 255], 'greyscale': 4},
    'barren_land': {'color': [255, 255, 255], 'greyscale': 5},
    'unknown': {'color': [0, 0, 0], 'greyscale': 6}
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

# Paths to the folders
mask_input_folder = '../data/land_cover/ann_dir/greytrain'
sat_input_folder = '../data/land_cover/img_dir/train'
mask_output_folder = '../data/land_cover/ann_dir/train'
sat_output_folder = '../data/land_cover/img_dir/train'
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
        mask = cv2.resize(mask, (256, 256))

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
