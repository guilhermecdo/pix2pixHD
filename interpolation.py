import cv2
import os
from pathlib import Path

# Paths
# Adjust these to your actual directory structure
input_dir = Path.home() / "git/pix2pixHD/CroppedImages/CroppedImages"
output_dir = Path.home() / "git/pix2pixHD/datasets/Oculos2DidsonInterpolation"

# Create target directories
path_a = output_dir / "train_A"
path_b = output_dir / "train_B"
path_a.mkdir(parents=True, exist_ok=True)
path_b.mkdir(parents=True, exist_ok=True)

# Target Size (Must be multiple of 32 for pix2pixHD, 256 is safe)
TARGET_SIZE = (256, 256)

def process_and_organize():
    files = os.listdir(input_dir)
    count = 0

    print(f"Starting interpolation (Lanczos4) for images in {input_dir}")

    for file_name in files:
        # Check for the Oculus files (input)
        if file_name.startswith('O') and file_name.endswith('.jpg'):
            # Find matching Didson file (ground truth)
            didson_name = 'D' + file_name[1:]
            
            oculus_path = input_dir / file_name
            didson_path = input_dir / didson_name

            if didson_path.exists():
                # Read images
                img_oculus = cv2.imread(str(oculus_path))
                img_didson = cv2.imread(str(didson_path))

                # Apply Lanczos4 Interpolation
                # dsize is (width, height)
                res_oculus = cv2.resize(img_oculus, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
                res_didson = cv2.resize(img_didson, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

                # Save with common name (removing 'O' and 'D' prefixes)
                common_name = file_name[1:]
                cv2.imwrite(str(path_a / common_name), res_oculus)
                cv2.imwrite(str(path_b / common_name), res_didson)
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} pairs...")

    print(f"Finished! Processed and interpolated {count} pairs to {TARGET_SIZE}")

if __name__ == "__main__":
    process_and_organize()