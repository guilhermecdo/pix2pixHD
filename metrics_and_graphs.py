import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# Native pix2pixHD components
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util


def get_foreground_mask(session, img_rgb):
    """Extract binary foreground mask using rembg."""
    from rembg import remove
    rgba = remove(img_rgb, session=session)
    alpha = rgba[:, :, 3]
    return (alpha > 10).astype(np.uint8)


def to_lpips_tensor(crop_bgr, device):
    """Convert a BGR/Grayscale crop array [H, W, (C)] to an LPIPS PyTorch tensor [-1, 1] [1, 3, H, W]."""
    if len(crop_bgr.shape) == 2:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = crop_rgb.shape[:2]
    if h < 32 or w < 32:
        crop_rgb = cv2.resize(crop_rgb, (max(w, 32), max(h, 32)), interpolation=cv2.INTER_LINEAR)
    
    t = torch.from_numpy(crop_rgb).float() / 127.5 - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def evaluate():
    parser = TestOptions()
    # Add custom rembg toggle argument
    parser.parser.add_argument("--no_rembg", action="store_true",
                               help="Disable background removal with rembg and evaluate whole image")
    opt = parser.parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    use_rembg = not opt.no_rembg
    mode_name = "rembg" if use_rembg else "full_frame"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(f"./metrics_evaluation_{mode_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Background Removal: {'Enabled (Foreground Only)' if use_rembg else 'Disabled (Full Frame)'}")

    # 1. Initialize LPIPS metric model (AlexNet)
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)
    loss_fn_vgg.eval()

    # 2. Check for checkpoints
    weights_path = Path(opt.checkpoints_dir) / opt.name
    if not weights_path.exists():
        weights_path = Path(opt.checkpoints_dir)
        
    weight_files = list(weights_path.glob("*_net_G.pth"))
    epoch_weight_map = {}
    for wf in weight_files:
        match = re.match(r"^(\d+)_net_G\.pth$", wf.name)
        if match:
            epoch_weight_map[int(match.group(1))] = wf

    epochs = sorted(epoch_weight_map.keys())
    if not epochs:
        raise FileNotFoundError(f"No epoch weights matching '<epoch>_net_G.pth' found in {weights_path}")
    print(f"Found {len(epochs)} epoch checkpoints to evaluate: {epochs}")

    # 3. Initialize DataLoader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f"Loaded dataset containing {len(dataset)} evaluation samples.")

    # 4. Cache evaluation samples (with or without rembg mask)
    rembg_session = None
    if use_rembg:
        from rembg import new_session
        print("Pre-processing Ground Truth images and computing foreground masks with rembg...")
        rembg_session = new_session("u2net")
    else:
        print("Pre-caching dataset images for full-frame evaluation...")
    
    cached_samples = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        real_im = util.tensor2im(data['image'][0])

        if len(real_im.shape) == 3 and real_im.shape[2] == 3:
            real_gray = cv2.cvtColor(real_im, cv2.COLOR_RGB2GRAY)
        else:
            real_gray = real_im.copy()

        if use_rembg:
            mask = get_foreground_mask(rembg_session, real_im)
            real_eval = cv2.bitwise_and(real_gray, real_gray, mask=mask)
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                bbox = (np.min(y_indices), np.max(y_indices), np.min(x_indices), np.max(x_indices))
            else:
                bbox = (0, real_gray.shape[0], 0, real_gray.shape[1])
        else:
            mask = None
            real_eval = real_gray
            bbox = (0, real_gray.shape[0], 0, real_gray.shape[1])

        cached_samples.append({
            'label': data['label'],
            'inst': data['inst'],
            'image': data['image'],
            'mask': mask,
            'real_eval': real_eval,
            'bbox': bbox
        })

    print(f"Caching complete for {len(cached_samples)} test samples.")

    stats = {
        'ssim_mean': [], 'ssim_std': [],
        'psnr_mean': [], 'psnr_std': [],
        'lpips_mean': [], 'lpips_std': []
    }

    # 5. Checkpoint evaluation loop
    for ep in epochs:
        print(f"Evaluating Epoch {ep:3d}...", end="", flush=True)
        opt.which_epoch = str(ep)
        
        model = create_model(opt)
        ssim_vals, psnr_vals, lpips_vals = [], [], []

        with torch.no_grad():
            for sample in cached_samples:
                generated = model.inference(sample['label'], sample['inst'], sample['image'])
                fake_im = util.tensor2im(generated.data[0])

                if len(fake_im.shape) == 3 and fake_im.shape[2] == 3:
                    fake_gray = cv2.cvtColor(fake_im, cv2.COLOR_RGB2GRAY)
                else:
                    fake_gray = fake_im.copy()

                real_eval = sample['real_eval']
                if fake_gray.shape != real_eval.shape:
                    fake_gray = cv2.resize(fake_gray, (real_eval.shape[1], real_eval.shape[0]))

                if use_rembg:
                    mask = sample['mask']
                    ymin, ymax, xmin, xmax = sample['bbox']
                    fake_eval = cv2.bitwise_and(fake_gray, fake_gray, mask=mask)

                    crop_fake = fake_eval[ymin:ymax+1, xmin:xmax+1]
                    crop_real = real_eval[ymin:ymax+1, xmin:xmax+1]

                    if crop_fake.shape[0] < 7 or crop_fake.shape[1] < 7:
                        crop_fake, crop_real = fake_eval, real_eval
                else:
                    crop_fake = fake_gray
                    crop_real = real_eval

                # SSIM
                s_val = ssim(crop_fake, crop_real, data_range=255)
                ssim_vals.append(s_val)

                # PSNR
                p_val = psnr(crop_real, crop_fake, data_range=255)
                psnr_vals.append(p_val)

                # LPIPS
                t_fake = to_lpips_tensor(crop_fake, device)
                t_real = to_lpips_tensor(crop_real, device)
                l_val = loss_fn_vgg(t_fake, t_real).item()
                lpips_vals.append(l_val)

        stats['ssim_mean'].append(float(np.mean(ssim_vals)))
        stats['ssim_std'].append(float(np.std(ssim_vals)))
        stats['psnr_mean'].append(float(np.mean(psnr_vals)))
        stats['psnr_std'].append(float(np.std(psnr_vals)))
        stats['lpips_mean'].append(float(np.mean(lpips_vals)))
        stats['lpips_std'].append(float(np.std(lpips_vals)))

        print(f" SSIM: {stats['ssim_mean'][-1]:.3f}±{stats['ssim_std'][-1]:.3f} | "
              f"PSNR: {stats['psnr_mean'][-1]:.2f}±{stats['psnr_std'][-1]:.2f} dB | "
              f"LPIPS: {stats['lpips_mean'][-1]:.3f}±{stats['lpips_std'][-1]:.3f}")

    for k in stats:
        stats[k] = np.array(stats[k])

    # 6. Save text log
    txt_path = output_dir / f"{opt.name}_{mode_name}_metrics_progression.txt"
    with open(txt_path, "w") as f:
        f.write("Epoch\tSSIM_Mean\tSSIM_Std\tPSNR_Mean\tPSNR_Std\tLPIPS_Mean\tLPIPS_Std\n")
        for idx, ep in enumerate(epochs):
            f.write(f"{ep}\t{stats['ssim_mean'][idx]:.6f}\t{stats['ssim_std'][idx]:.6f}\t"
                    f"{stats['psnr_mean'][idx]:.6f}\t{stats['psnr_std'][idx]:.6f}\t"
                    f"{stats['lpips_mean'][idx]:.6f}\t{stats['lpips_std'][idx]:.6f}\n")
    print(f"\nSaved metric log: {txt_path}")

    # 7. Generate 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    label_prefix = "Foreground" if use_rembg else "Full"

    # Plot SSIM
    axes[0].plot(epochs, stats['ssim_mean'], marker='o', color='#1f77b4', lw=2, label='Mean SSIM')
    axes[0].fill_between(epochs, np.clip(stats['ssim_mean'] - stats['ssim_std'], 0, 1),
                         np.clip(stats['ssim_mean'] + stats['ssim_std'], 0, 1), color='#1f77b4', alpha=0.2)
    axes[0].set_title(f"{label_prefix} SSIM (Higher is better)", fontweight='bold')
    axes[0].set_ylabel("SSIM")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Plot PSNR
    axes[1].plot(epochs, stats['psnr_mean'], marker='s', color='#2ca02c', lw=2, label='Mean PSNR')
    axes[1].fill_between(epochs, stats['psnr_mean'] - stats['psnr_std'],
                         stats['psnr_mean'] + stats['psnr_std'], color='#2ca02c', alpha=0.2)
    axes[1].set_title(f"{label_prefix} PSNR (Higher is better)", fontweight='bold')
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Plot LPIPS
    axes[2].plot(epochs, stats['lpips_mean'], marker='^', color='#d62728', lw=2, label='Mean LPIPS')
    axes[2].fill_between(epochs, np.clip(stats['lpips_mean'] - stats['lpips_std'], 0, None),
                         stats['lpips_mean'] + stats['lpips_std'], color='#d62728', alpha=0.2)
    axes[2].set_title(f"{label_prefix} LPIPS (Lower is better)", fontweight='bold')
    axes[2].set_ylabel("LPIPS Distance")
    axes[2].grid(True, linestyle='--', alpha=0.5)

    step = max(1, len(epochs) // 10)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs[::step])
        ax.legend(loc='best')

    plt.suptitle(f"Transfer/Reconstruction Metrics Progression ({mode_name}) — {opt.name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_path = output_dir / f"{opt.name}_{mode_name}_metrics_panel.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved metric chart: {chart_path}")


if __name__ == "__main__":
    evaluate()