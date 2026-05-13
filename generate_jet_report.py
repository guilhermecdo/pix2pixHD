import cv2
import os
from pathlib import Path

def generate_jet_report(results_dir, output_dir):
    res_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    images = list(res_path.glob("*.jpg"))
    
    for img_p in images:
        # Load as grayscale
        gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        
        # Apply JET Color Map
        color_jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Save to new report folder
        cv2.imwrite(str(out_path / img_p.name), color_jet)

if __name__ == "__main__":
    generate_jet_report("./results/Exp_pad_256_p0.9/test_latest/images", "./results/Report_JET")