import argparse
import cv2
import os
import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to generated images")
    return parser.parse_args()

def load_image_for_lpips(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return (img / 127.5) - 1.0

def calculate_metrics(results_dir):
    res_path = Path(results_dir).resolve()
    
    # 1. Find all synthesized images
    fake_images = list(res_path.glob("*_synthesized_image.jpg"))
    
    if not fake_images:
        print(f"ERROR: No synthesized images found in {res_path}")
        return

    # Use GPU for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(device)
    
    all_mse, all_psnr, all_ssim, all_lpips = [], [], [], []

    for fake_p in fake_images:
        # 2. Match with the Ground Truth (_input_label.jpg)
        # Based on your output: obj_1_Didson104_Oculus104_input_label
        real_p_str = str(fake_p).replace("_synthesized_image.jpg", "_input_label.jpg")
        real_p = Path(real_p_str)
        
        if not real_p.exists():
            continue

        img_real = cv2.imread(str(real_p), cv2.IMREAD_GRAYSCALE)
        img_fake = cv2.imread(str(fake_p), cv2.IMREAD_GRAYSCALE)

        if img_real is None or img_fake is None:
            continue

        # Calculate standard metrics
        all_mse.append(np.mean((img_real - img_fake) ** 2))
        all_psnr.append(psnr(img_real, img_fake, data_range=255))
        all_ssim.append(ssim(img_real, img_fake, data_range=255))

        # Perceptual metric
        t_real = load_image_for_lpips(real_p).to(device)
        t_fake = load_image_for_lpips(fake_p).to(device)
        all_lpips.append(loss_fn_vgg(t_real, t_fake).item())

    if len(all_mse) == 0:
        print(f"ERROR: Found synthesized images but could not find matching _input_label images.")
        return

    # Extract Experiment Name
    try:
        exp_name = res_path.parts[-3]
    except:
        exp_name = "Unknown_Exp"

    output = f"EXP: {exp_name} | Samples: {len(all_mse)} | MSE: {np.mean(all_mse):.4f} | PSNR: {np.mean(all_psnr):.2f} | SSIM: {np.mean(all_ssim):.4f} | LPIPS: {np.mean(all_lpips):.4f}"
    print(output)

if __name__ == "__main__":
    args = get_args()
    calculate_metrics(args.results_dir)