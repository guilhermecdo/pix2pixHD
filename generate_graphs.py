import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from rembg import remove, new_session

# Native pix2pixHD components
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util


def get_foreground_mask(session, img_rgb):
    """Extract binary foreground mask using rembg."""
    rgba = remove(img_rgb, session=session)
    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8)
    return mask


def evaluate():
    # 1. Parse standard pix2pixHD test options
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    output_dir = Path("./ssim_evaluation_rembg")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Check for checkpoints
    weights_path = Path(opt.checkpoints_dir) / opt.name
    if not weights_path.exists():
        weights_path = Path(opt.checkpoints_dir)
        
    weight_files = list(weights_path.glob("*_net_G.pth"))

    epoch_weight_map = {}
    for wf in weight_files:
        match = re.match(r"^(\d+)_net_G\.pth$", wf.name)
        if match:
            epoch = int(match.group(1))
            epoch_weight_map[epoch] = wf

    epochs = sorted(epoch_weight_map.keys())
    if not epochs:
        raise FileNotFoundError(f"No epoch weights matching '<epoch>_net_G.pth' found in {weights_path}")
    print(f"Found {len(epochs)} epoch checkpoints to evaluate: {epochs}")

    # 3. Initialize DataLoader once
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f"Loaded dataset containing {len(dataset)} evaluation samples.")

    # 4. Pre-cache ground truth images and extract rembg masks
    print("Pre-processing Ground Truth images and computing foreground masks with rembg...")
    rembg_session = new_session("u2net")
    
    cached_samples = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        real_im = util.tensor2im(data['image'][0])
        mask = get_foreground_mask(rembg_session, real_im)

        # Grayscale version of real image
        if len(real_im.shape) == 3 and real_im.shape[2] == 3:
            real_gray = cv2.cvtColor(real_im, cv2.COLOR_RGB2GRAY)
        else:
            real_gray = real_im.copy()

        # Apply mask
        real_masked = cv2.bitwise_and(real_gray, real_gray, mask=mask)

        # Bounding box of foreground to evaluate SSIM tightly on object
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            bbox = (np.min(y_indices), np.max(y_indices), np.min(x_indices), np.max(x_indices))
        else:
            bbox = (0, real_gray.shape[0], 0, real_gray.shape[1])

        cached_samples.append({
            'label': data['label'],
            'inst': data['inst'],
            'image': data['image'],
            'mask': mask,
            'real_masked': real_masked,
            'bbox': bbox
        })

    print(f"Mask caching complete for {len(cached_samples)} test samples.")

    epoch_mean_ssim = []
    epoch_std_ssim = []

    # 5. Iterate over each epoch checkpoint
    for ep in epochs:
        print(f"Evaluating Epoch {ep:3d}...", end="", flush=True)
        opt.which_epoch = str(ep)
        
        model = create_model(opt)
        ssim_vals = []

        with torch.no_grad():
            for sample in cached_samples:
                # Inference
                generated = model.inference(sample['label'], sample['inst'], sample['image'])
                fake_im = util.tensor2im(generated.data[0])

                if len(fake_im.shape) == 3 and fake_im.shape[2] == 3:
                    fake_gray = cv2.cvtColor(fake_im, cv2.COLOR_RGB2GRAY)
                else:
                    fake_gray = fake_im.copy()

                mask = sample['mask']
                real_masked = sample['real_masked']
                ymin, ymax, xmin, xmax = sample['bbox']

                # Resize fake if shape doesn't match real
                if fake_gray.shape != real_masked.shape:
                    fake_gray = cv2.resize(fake_gray, (real_masked.shape[1], real_masked.shape[0]))

                # Apply the same foreground mask to the synthesized image
                fake_masked = cv2.bitwise_and(fake_gray, fake_gray, mask=mask)

                # Crop to foreground bounding box
                crop_fake = fake_masked[ymin:ymax+1, xmin:xmax+1]
                crop_real = real_masked[ymin:ymax+1, xmin:xmax+1]

                if crop_fake.shape[0] < 7 or crop_fake.shape[1] < 7:
                    # Minimum window size requirement for skimage SSIM is 7x7
                    val = ssim(fake_masked, real_masked, data_range=255)
                else:
                    val = ssim(crop_fake, crop_real, data_range=255)

                ssim_vals.append(val)

        mean_val = float(np.mean(ssim_vals)) if ssim_vals else 0.0
        std_val = float(np.std(ssim_vals)) if ssim_vals else 0.0

        epoch_mean_ssim.append(mean_val)
        epoch_std_ssim.append(std_val)
        print(f" Mean SSIM: {mean_val:.4f} ± {std_val:.4f}")

    epoch_mean_ssim = np.array(epoch_mean_ssim)
    epoch_std_ssim = np.array(epoch_std_ssim)

    # 6. Save text record (Epoch, Mean, Std)
    txt_path = output_dir / f"{opt.name}_foreground_ssim_progression.txt"
    with open(txt_path, "w") as f:
        f.write("Epoch\tMean_SSIM\tStd_SSIM\n")
        for ep, mean_v, std_v in zip(epochs, epoch_mean_ssim, epoch_std_ssim):
            f.write(f"{ep}\t{mean_v:.6f}\t{std_v:.6f}\n")
    print(f"\nSaved textual SSIM records to: {txt_path}")

    # 7. Plot and save metric chart with shaded std band
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(epochs, epoch_mean_ssim, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Mean Foreground SSIM')
    
    # Fill between ± 1 standard deviation
    upper_bound = np.clip(epoch_mean_ssim + epoch_std_ssim, 0.0, 1.0)
    lower_bound = np.clip(epoch_mean_ssim - epoch_std_ssim, 0.0, 1.0)
    plt.fill_between(epochs, lower_bound, upper_bound, color='#1f77b4', alpha=0.2, label='±1 Std Dev')

    plt.title(f"Acoustic-to-Optical Foreground SSIM ({opt.name})", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("SSIM", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(epochs[::max(1, len(epochs)//15)])
    
    y_min = max(0.0, float(np.min(lower_bound)) - 0.05)
    y_max = min(1.0, float(np.max(upper_bound)) + 0.05)
    plt.ylim(y_min, y_max)
    plt.legend(loc='lower right')
    plt.tight_layout()

    chart_path = output_dir / f"{opt.name}_foreground_ssim_vs_epochs.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved metric chart to: {chart_path}")


if __name__ == "__main__":
    evaluate()