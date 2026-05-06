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
    parser.add_argument("--mode", type=str, choices=['pad', 'interp'], required=True, help="How the data was prepared")
    parser.add_argument("--orig_w", type=int, default=70, help="Original width before processing")
    parser.add_argument("--orig_h", type=int, default=40, help="Original height before processing")
    return parser.parse_args()

def get_content_crop(img_gray):
    """Finds the bounding box of non-black pixels to remove padding."""
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h
    return None

def calculate_metrics_original(results_dir, mode, orig_w, orig_h):
    res_path = Path(results_dir).resolve()
    fake_images = list(res_path.glob("*_synthesized_image.jpg"))
    
    if not fake_images:
        print(f"ERROR: No images found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(device)
    
    all_mse, all_psnr, all_ssim, all_lpips = [], [], [], []

    for fake_p in fake_images:
        real_p = Path(str(fake_p).replace("_synthesized_image.jpg", "_input_label.jpg"))
        if not real_p.exists(): continue

        # Load images
        img_real = cv2.imread(str(real_p))
        img_fake = cv2.imread(str(fake_p))
        
        # Convert to gray for crop detection and traditional metrics
        gray_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
        gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)

        if mode == 'pad':
            # Remove padding by finding the actual sonar content
            crop_box = get_content_crop(gray_real)
            if crop_box:
                x, y, w, h = crop_box
                eval_real = gray_real[y:y+h, x:x+w]
                eval_fake = gray_fake[y:y+h, x:x+w]
                # For LPIPS (perceptual), we crop the color versions
                lpips_real = img_real[y:y+h, x:x+w]
                lpips_fake = img_fake[y:y+h, x:x+w]
            else:
                continue
        else: # mode == 'interp'
            # Downsample back to original sonar resolution (70x40)
            eval_real = cv2.resize(gray_real, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            eval_fake = cv2.resize(gray_fake, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            lpips_real = cv2.resize(img_real, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
            lpips_fake = cv2.resize(img_fake, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # Calculate metrics on the "Original Form"
        all_mse.append(np.mean((eval_real.astype(float) - eval_fake.astype(float)) ** 2))
        all_psnr.append(psnr(eval_real, eval_fake, data_range=255))
        all_ssim.append(ssim(eval_real, eval_fake, data_range=255))

        # LPIPS needs tensors
        t_real = (torch.from_numpy(lpips_real.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 127.5) - 1.0
        t_fake = (torch.from_numpy(lpips_fake.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 127.5) - 1.0
        all_lpips.append(loss_fn_vgg(t_real, t_fake).item())

    exp_name = res_path.parts[-3]
    print(f"EXP (ORIGINAL FORM): {exp_name} | Samples: {len(all_mse)} | MSE: {np.mean(all_mse):.4f} | PSNR: {np.mean(all_psnr):.2f} | SSIM: {np.mean(all_ssim):.4f} | LPIPS: {np.mean(all_lpips):.4f}")

if __name__ == "__main__":
    args = get_args()
    calculate_metrics_original(args.results_dir, args.mode, args.orig_w, args.orig_h)