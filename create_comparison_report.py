import cv2
import numpy as np
from pathlib import Path
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()

def create_comparison_report(results_dir, output_dir):
    res_path = Path(results_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    # pix2pixHD naming logic for your config:
    # Synthesized: name_synthesized_image.jpg
    # GT (Didson): name_input_label.jpg
    # Input (Oculus): name_real_image.jpg
    
    fakes = list(res_path.glob("*_synthesized_image.jpg"))
    print(f"Found {len(fakes)} images to process...")

    for fake_p in fakes:
        base = str(fake_p).replace("_synthesized_image.jpg", "")
        real_p = Path(base + "_input_label.jpg") # Ground Truth Didson
        input_p = Path(base + "_real_image.jpg") # Input Oculus
        
        # Check if all files exist
        if not (real_p.exists() and input_p.exists()):
            continue

        img_input = cv2.imread(str(input_p))
        img_fake = cv2.imread(str(fake_p))
        img_real = cv2.imread(str(real_p))
        
        # Verify images loaded correctly
        if img_input is None or img_fake is None or img_real is None:
            continue
            
        # Tile them: [Oculus | Generated | Didson GT]
        comparison = np.hstack((img_input, img_fake, img_real))
        
        # Draw Divider lines (optional, helps separate the frames)
        h, w, _ = img_input.shape
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        cv2.line(comparison, (w*2, 0), (w*2, h), (255, 255, 255), 2)
        
        cv2.imwrite(str(out_path / fake_p.name), comparison)

if __name__ == "__main__":
    args = get_args()
    create_comparison_report(args.results_dir, args.output_dir)