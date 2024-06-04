import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the image file
image_path = "../data/land_cover/ann_dir/train/470446_mask.png"

# Define the color mapping based on the provided order
color_mapping = {
    'urban_land': {'color': [0, 255, 255], 'greyscale': 0},
    'agriculture_land': {'color': [255, 255, 0], 'greyscale': 1},
    'rangeland': {'color': [255, 0, 255], 'greyscale': 2},
    'forest_land': {'color': [0, 255, 0], 'greyscale': 3},
    'water': {'color': [0, 0, 255], 'greyscale': 4},
    'barren_land': {'color': [255, 255, 255], 'greyscale': 5},
    'unknown': {'color': [0, 0, 0], 'greyscale': 6}
}

# Create a palette based on the color mapping
palette = np.array([color_mapping[key]['color'] for key in color_mapping])

# Read the image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Map each pixel value to the corresponding color in the palette
colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
for key in color_mapping:
    color = color_mapping[key]['color']
    greyscale_value = color_mapping[key]['greyscale']
    colored_img[img == greyscale_value] = color

# Display the image
plt.imshow(colored_img)
plt.axis('off')  # Hide the axis
plt.show()

# Print unique values in the original mask
print("Unique values in the mask:", np.unique(img))
