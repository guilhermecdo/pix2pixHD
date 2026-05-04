import os
import shutil
from pathlib import Path

# Configuration
source_dir = Path.home() / "git/pix2pixHD/CroppedImages/CroppedImages"
target_base = Path.home() / "git/pix2pixHD/datasets/Oculus2Didson"

def organize_pix2pix():
    # Define paths
    path_a = target_base / "train_A"  # Input (Oculus)
    path_b = target_base / "train_B"  # Ground Truth (Didson)

    # Create directories if they don't exist
    path_a.mkdir(parents=True, exist_ok=True)
    path_b.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {source_dir}")
    
    files = os.listdir(source_dir)
    count = 0

    for file_name in files:
        # We only care about the Oculus files to start the pairing
        if file_name.startswith('O') and file_name.endswith('.jpg'):
            # The corresponding Didson filename replaces 'O' with 'D'
            didson_name = 'D' + file_name[1:]
            
            oculus_path = source_dir / file_name
            didson_path = source_dir / didson_name

            # Check if the pair exists
            if didson_path.exists():
                # We rename them to a common name so they match in both folders
                # e.g., obj_1_Didson100_Oculus100.jpg
                common_name = file_name[1:] 
                
                shutil.copy(oculus_path, path_a / common_name)
                shutil.copy(didson_path, path_b / common_name)
                count += 1
            else:
                print(f"Warning: Match not found for {file_name}")

    print(f"Finished! Successfully moved and paired {count} images.")
    print(f"Data is now at: {target_base}")

if __name__ == "__main__":
    organize_pix2pix()