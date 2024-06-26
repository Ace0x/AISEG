import mmcv
import os
from mmseg.apis import inference_model, init_model
import numpy as np
import matplotlib.pyplot as plt

# Configuration and checkpoint files
config_file = '../configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_LandClass-512x1024.py'
checkpoint_file = '../work_dirs/deeplabv3plus_r50-d8_4xb2-40k_LandClass-512x1024/iter_40000.pth'

# Initialize the segmentor
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Custom palette
palette = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]

# Convert palette to numpy array for easy indexing
palette = np.array(palette, dtype=np.uint8)

# Directory containing the image(s) for inference
image_dir = '../data/land_cover/train/testv2/'

# Directory to save the results
output_dir = '../results/deeplabv3plus/'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all images in the directory
for image_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, image_name)
    
    # Run inference
    result = inference_model(model, img_path)
    
    # Extract the predicted mask from SegDataSample
    result_mask = result.pred_sem_seg.data.cpu().numpy().astype(np.uint8)
    
    # Debug: Print shape and type of result_mask
    print(f"Processing {image_name}: result_mask shape {result_mask.shape}, dtype {result_mask.dtype}")
    
    # Remove the extra dimension from result_mask
    if result_mask.shape[0] == 1:
        result_mask = result_mask.squeeze(0)
    
    # Check if the result_mask contains expected class values
    unique_classes = np.unique(result_mask)
    print(f"Processing {image_name}: unique classes in result_mask {unique_classes}")
    
    # Ensure the result_mask contains only values within the expected range
    result_mask = np.clip(result_mask, 0, len(palette) - 1)

    # Map the single-channel mask to a 3-channel image using the palette
    color_mask = palette[result_mask]
    
    # Debug: Print shape and type of color_mask
    print(f"Processing {image_name}: color_mask shape {color_mask.shape}, dtype {color_mask.dtype}")

    # Visualize the result mask and color mask
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(result_mask, cmap='gray')
    plt.title('Result Mask (Greyscale)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(color_mask)
    plt.title('Color Mask')
    
    # Load the expected mask for comparison
    expected_mask_path = f"../data/land_cover/mask_dir/val/{image_name.replace('.jpg', '.png')}"
    if os.path.exists(expected_mask_path):
        expected_mask = mmcv.imread(expected_mask_path)
        plt.subplot(1, 3, 3)
        plt.imshow(expected_mask)
        plt.title('Expected Mask')
    
    
    # Save the color mask image using PIL to ensure correct format
    from PIL import Image
    output_path = os.path.join(output_dir, image_name)
    Image.fromarray(color_mask).save(output_path)

print('Inference completed and results saved to:', output_dir)
