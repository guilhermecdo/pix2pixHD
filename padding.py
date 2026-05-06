import cv2
import os
import numpy as np
from pathlib import Path

# Configuration
dataset_path = Path.home() / "git/pix2pixHD/datasets/Oculus2DidsonPad"
target_size = 256  # Must be a power of 2 (128, 256, 512)

def pad_dataset():
    for folder in ['train_A', 'train_B']:
        folder_path = dataset_path / folder
        if not folder_path.exists():
            print(f"Skipping {folder}, path not found.")
            continue

        print(f"Padding images in {folder}...")
        
        for img_name in os.listdir(folder_path):
            img_path = folder_path / img_name
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue

            h, w = img.shape[:2]
            
            # Calculate padding to center the image
            top = (target_size - h) // 2
            bottom = target_size - h - top
            left = (target_size - w) // 2
            right = target_size - w - left

            # Add black border (constant value 0)
            padded_img = cv2.copyMakeBorder(
                img, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # Overwrite the original image with the padded version
            cv2.imwrite(str(img_path), padded_img)

    print(f"Finished padding all images to {target_size}x{target_size}.")

if __name__ == "__main__":
    pad_dataset()