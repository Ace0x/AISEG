import os
import shutil
from sklearn.model_selection import train_test_split

# Define the directories
images_dir = '../data/land_cover/train/sat'
masks_dir = '../data/land_cover/train/mask'
train_images_dir = '../data/land_cover/img_dir/train'
train_masks_dir = '../data/land_cover/ann_dir/train'
val_images_dir = '../data/land_cover/img_dir/val'
val_masks_dir = '../data/land_cover/ann_dir/val'

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

# List all files in the image and mask directories
image_files = sorted([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
mask_files = sorted([f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))])

# Ensure the image files correspond to the mask files
assert len(image_files) == len(mask_files), "The number of images and masks must be the same"

# Split the files into training and validation sets (80/20 split)
train_images, val_images, train_masks, val_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)

# Function to copy files to the new directory
def copy_files(files, src_dir, dst_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

# Copy the training files
copy_files(train_images, images_dir, train_images_dir)
copy_files(train_masks, masks_dir, train_masks_dir)

# Copy the validation files
copy_files(val_images, images_dir, val_images_dir)
copy_files(val_masks, masks_dir, val_masks_dir)

print(f"Training set: {len(train_images)} images, {len(train_masks)} masks")
print(f"Validation set: {len(val_images)} images, {len(val_masks)} masks")
