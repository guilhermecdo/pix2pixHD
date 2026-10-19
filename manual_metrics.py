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


def extract_id(filename_or_path):
    """
    Extract key identifier or trailing numeric sequence.
    Example: 'Sonar_0044' -> '0044', 'RGB_0044.png' -> '0044'
    """
    stem = Path(filename_or_path).stem
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return numbers[-1]
    for prefix in ["Sonar_", "Son_", "RGB_", "Optical_", "Real_"]:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def extract_binary_mask(img_bgr_or_bgra):
    """Extract binary mask from non-black/non-zero pixels of the manual image."""
    if len(img_bgr_or_bgra.shape) == 3 and img_bgr_or_bgra.shape[2] == 4:
        return (img_bgr_or_bgra[:, :, 3] > 0).astype(np.uint8)
    elif len(img_bgr_or_bgra.shape) == 3 and img_bgr_or_bgra.shape[2] >= 3:
        return (np.any(img_bgr_or_bgra[:, :, :3] > 0, axis=-1)).astype(np.uint8)
    else:
        return (img_bgr_or_bgra > 0).astype(np.uint8)


def to_lpips_tensor(crop_rgb, device):
    """Convert an RGB crop array [H, W, 3] to an LPIPS PyTorch tensor [-1, 1] [1, 3, H, W]."""
    h, w = crop_rgb.shape[:2]
    if h < 32 or w < 32:
        crop_rgb = cv2.resize(crop_rgb, (max(w, 32), max(h, 32)), interpolation=cv2.INTER_LINEAR)
        
    t = torch.from_numpy(crop_rgb).float() / 127.5 - 1.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def evaluate():
    parser = TestOptions()
    parser.parser.add_argument("--remove_back_dir", type=str, required=True,
                               help="Path to folder containing manual background removed target images")
    parser.parser.add_argument("--save_debug_images", action="store_true",
                               help="Save side-by-side diagnostic crops and masked images to disk")
    parser.parser.add_argument("--debug_epoch", type=int, default=None,
                               help="Specific epoch to export debug images for (defaults to the last epoch)")
    opt = parser.parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path("./metrics_manual_removeback")
    output_dir.mkdir(parents=True, exist_ok=True)

    remove_back_path = Path(opt.remove_back_dir).resolve()
    if not remove_back_path.exists():
        raise FileNotFoundError(f"Folder '{remove_back_path}' does not exist.")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    manual_files_by_stem = {}
    manual_files_by_id = {}
    
    for f in remove_back_path.iterdir():
        if f.suffix.lower() in valid_exts:
            manual_files_by_stem[f.stem] = f
            num_id = extract_id(f.name)
            manual_files_by_id[num_id] = f

    print(f"Found {len(manual_files_by_stem)} manual ground truth images in {remove_back_path}")

    # 1. Initialize LPIPS perceptual loss model (AlexNet)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_alex.eval()

    # 2. Find epoch checkpoint weights
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

    # Determine which epoch will save visual inspection images
    debug_epoch_target = opt.debug_epoch if opt.debug_epoch is not None else epochs[-1]
    debug_save_dir = output_dir / f"debug_visuals_epoch_{debug_epoch_target}"
    if opt.save_debug_images:
        debug_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visual debugging enabled: saving image frames to {debug_save_dir} on Epoch {debug_epoch_target}")

    # 3. Initialize DataLoader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print(f"Loaded dataset containing {len(dataset)} evaluation samples.")

    # 4. Pre-cache manual foreground targets and extract masks
    print("Matching test samples with 'remove_back' masks...")
    cached_samples = []

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        path_str = ""
        if 'B_paths' in data and data['B_paths']:
            path_str = data['B_paths'][0]
        elif 'path' in data and data['path']:
            path_str = data['path'][0]

        stem = Path(path_str).stem
        num_id = extract_id(path_str)

        matched_manual_file = None
        if stem in manual_files_by_stem:
            matched_manual_file = manual_files_by_stem[stem]
        elif num_id in manual_files_by_id:
            matched_manual_file = manual_files_by_id[num_id]
        else:
            for m_key, m_path in manual_files_by_stem.items():
                if num_id in m_key or stem in m_key or m_key in stem:
                    matched_manual_file = m_path
                    break

        if matched_manual_file is None:
            print(f"Warning: No matching mask found in remove_back for sample '{stem}' (ID: '{num_id}'). Skipping.")
            continue

        manual_img = cv2.imread(str(matched_manual_file), cv2.IMREAD_UNCHANGED)
        if manual_img is None:
            continue

        mask = extract_binary_mask(manual_img)

        if len(manual_img.shape) == 3 and manual_img.shape[2] == 4:
            manual_rgb = cv2.cvtColor(manual_img, cv2.COLOR_BGRA2RGB)
        elif len(manual_img.shape) == 3 and manual_img.shape[2] == 3:
            manual_rgb = cv2.cvtColor(manual_img, cv2.COLOR_BGR2RGB)
        else:
            manual_rgb = cv2.cvtColor(manual_img, cv2.COLOR_GRAY2RGB)

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        manual_masked_rgb = np.where(mask_3d > 0, manual_rgb, 0).astype(np.uint8)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            bbox = (np.min(y_indices), np.max(y_indices), np.min(x_indices), np.max(x_indices))
        else:
            bbox = (0, manual_rgb.shape[0], 0, manual_rgb.shape[1])

        # Get raw sonar input visualization for side-by-side debugging
        sonar_rgb = util.tensor2im(data['label'][0]) if 'label' in data else None

        cached_samples.append({
            'label': data['label'],
            'inst': data['inst'],
            'image': data['image'],
            'mask': mask,
            'mask_3d': mask_3d,
            'manual_masked_rgb': manual_masked_rgb,
            'bbox': bbox,
            'stem': stem,
            'id': num_id,
            'sonar_rgb': sonar_rgb
        })

    if not cached_samples:
        raise RuntimeError("No test images could be matched with the files in 'remove_back'. Check file naming.")
    print(f"Successfully matched and cached {len(cached_samples)} manual foreground RGB pairs.")

    stats = {
        'ssim_mean': [], 'ssim_std': [],
        'psnr_mean': [], 'psnr_std': [],
        'lpips_mean': [], 'lpips_std': []
    }
    raw_points = {'ssim': [], 'psnr': [], 'lpips': []}

    # 5. Checkpoint evaluation loop
    for ep in epochs:
        print(f"Evaluating Epoch {ep:3d}...", end="", flush=True)
        opt.which_epoch = str(ep)

        model = create_model(opt)
        ssim_vals, psnr_vals, lpips_vals = [], [], []

        with torch.no_grad():
            for sample in cached_samples:
                generated = model.inference(sample['label'], sample['inst'], sample['image'])
                fake_rgb = util.tensor2im(generated.data[0])

                manual_masked_rgb = sample['manual_masked_rgb']
                mask = sample['mask']
                mask_3d = sample['mask_3d']
                ymin, ymax, xmin, xmax = sample['bbox']

                if fake_rgb.shape[:2] != manual_masked_rgb.shape[:2]:
                    fake_rgb = cv2.resize(fake_rgb, (manual_masked_rgb.shape[1], manual_masked_rgb.shape[0]))

                # Mask out pixels outside the manual object mask
                fake_masked_rgb = np.where(mask_3d > 0, fake_rgb, 0).astype(np.uint8)

                crop_fake = fake_masked_rgb[ymin:ymax+1, xmin:xmax+1]
                crop_manual = manual_masked_rgb[ymin:ymax+1, xmin:xmax+1]

                if crop_fake.shape[0] < 7 or crop_fake.shape[1] < 7:
                    crop_fake = fake_masked_rgb
                    crop_manual = manual_masked_rgb

                # Metrics
                s_val = ssim(crop_fake, crop_manual, channel_axis=2, data_range=255)
                ssim_vals.append(s_val)

                p_val = psnr(crop_manual, crop_fake, data_range=255)
                psnr_vals.append(p_val)

                t_fake = to_lpips_tensor(crop_fake, device)
                t_manual = to_lpips_tensor(crop_manual, device)
                l_val = loss_fn_alex(t_fake, t_manual).item()
                lpips_vals.append(l_val)

                # -------------------------------------------------------------
                # Save Debug Frames
                # -------------------------------------------------------------
                if opt.save_debug_images and ep == debug_epoch_target:
                    h, w = manual_masked_rgb.shape[:2]
                    stem = sample['stem']
                    num_id = sample['id']

                    # Convert to BGR for OpenCV saving
                    bgr_gt = cv2.cvtColor(manual_masked_rgb, cv2.COLOR_RGB2BGR)
                    bgr_synth_raw = cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2BGR)
                    bgr_synth_masked = cv2.cvtColor(fake_masked_rgb, cv2.COLOR_RGB2BGR)
                    bgr_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    if sample['sonar_rgb'] is not None:
                        sonar_bgr = cv2.cvtColor(sample['sonar_rgb'], cv2.COLOR_RGB2BGR)
                        sonar_bgr = cv2.resize(sonar_bgr, (w, h))
                    else:
                        sonar_bgr = np.zeros((h, w, 3), dtype=np.uint8)

                    # Superposition overlap: Green = GT manual, Red = Synth masked
                    overlap_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    gray_gt = cv2.cvtColor(bgr_gt, cv2.COLOR_BGR2GRAY)
                    gray_synth = cv2.cvtColor(bgr_synth_masked, cv2.COLOR_BGR2GRAY)
                    overlap_vis[:, :, 1] = gray_gt      # Green
                    overlap_vis[:, :, 2] = gray_synth   # Red

                    # Save individual panels
                    prefix_name = f"{num_id}_{stem}"
                    cv2.imwrite(str(debug_save_dir / f"{prefix_name}_gt_manual.png"), bgr_gt)
                    cv2.imwrite(str(debug_save_dir / f"{prefix_name}_synth_raw.png"), bgr_synth_raw)
                    cv2.imwrite(str(debug_save_dir / f"{prefix_name}_synth_masked.png"), bgr_synth_masked)
                    cv2.imwrite(str(debug_save_dir / f"{prefix_name}_mask_vis.png"), bgr_mask)

                    # Build side-by-side diagnostic strip:
                    # [ Sonar Input | GT Manual | Synth Raw | Synth Masked | Red/Green Overlap ]
                    strip_w = 320
                    strip_h = 320
                    s_sonar = cv2.resize(sonar_bgr, (strip_w, strip_h))
                    s_gt = cv2.resize(bgr_gt, (strip_w, strip_h))
                    s_raw = cv2.resize(bgr_synth_raw, (strip_w, strip_h))
                    s_masked = cv2.resize(bgr_synth_masked, (strip_w, strip_h))
                    s_overlap = cv2.resize(overlap_vis, (strip_w, strip_h))

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(s_sonar, "Sonar Input", (10, 25), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(s_gt, "GT Manual", (10, 25), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(s_raw, "Synth Raw", (10, 25), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(s_masked, "Synth Masked", (10, 25), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(s_overlap, f"Overlap (Y=Match)", (10, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    composite = np.hstack((s_sonar, s_gt, s_raw, s_masked, s_overlap))
                    footer = np.zeros((40, composite.shape[1], 3), dtype=np.uint8)
                    metric_str = f"ID: {num_id} | RGB SSIM: {s_val:.4f} | PSNR: {p_val:.2f} dB | LPIPS: {l_val:.4f}"
                    cv2.putText(footer, metric_str, (15, 26), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    full_debug_image = np.vstack((composite, footer))

                    cv2.imwrite(str(debug_save_dir / f"{prefix_name}_side_by_side.png"), full_debug_image)

        raw_points['ssim'].append(ssim_vals)
        raw_points['psnr'].append(psnr_vals)
        raw_points['lpips'].append(lpips_vals)

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
    txt_path = output_dir / f"{opt.name}_removeback_rgb_metrics_progression.txt"
    with open(txt_path, "w") as f:
        f.write("Epoch\tSSIM_Mean\tSSIM_Std\tPSNR_Mean\tPSNR_Std\tLPIPS_Mean\tLPIPS_Std\n")
        for idx, ep in enumerate(epochs):
            f.write(f"{ep}\t{stats['ssim_mean'][idx]:.6f}\t{stats['ssim_std'][idx]:.6f}\t"
                    f"{stats['psnr_mean'][idx]:.6f}\t{stats['psnr_std'][idx]:.6f}\t"
                    f"{stats['lpips_mean'][idx]:.6f}\t{stats['lpips_std'][idx]:.6f}\n")
    print(f"\nSaved metric log: {txt_path}")

    # 7. Generate 3-panel scatter progression figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)

    epoch_diff = np.diff(epochs)
    min_step = float(np.min(epoch_diff)) if len(epoch_diff) > 0 else 1.0
    jitter_width = min_step * 0.15

    # SSIM Plot
    for ep, vals in zip(epochs, raw_points['ssim']):
        x_jittered = np.random.normal(loc=ep, scale=jitter_width, size=len(vals))
        axes[0].scatter(x_jittered, vals, color='#1f77b4', alpha=0.4, s=28, edgecolors='none')
    axes[0].plot(epochs, stats['ssim_mean'], marker='o', markersize=6, color='#0b3c61', lw=2.5, label='Mean SSIM')
    axes[0].scatter([], [], color='#1f77b4', alpha=0.5, s=28, label='Per-Image RGB SSIM')
    axes[0].set_title("Object RGB SSIM vs Manual GT (Higher is better)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("SSIM", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # PSNR Plot
    for ep, vals in zip(epochs, raw_points['psnr']):
        x_jittered = np.random.normal(loc=ep, scale=jitter_width, size=len(vals))
        axes[1].scatter(x_jittered, vals, color='#2ca02c', alpha=0.4, s=28, edgecolors='none')
    axes[1].plot(epochs, stats['psnr_mean'], marker='s', markersize=6, color='#135e13', lw=2.5, label='Mean PSNR')
    axes[1].scatter([], [], color='#2ca02c', alpha=0.5, s=28, label='Per-Image RGB PSNR')
    axes[1].set_title("Object RGB PSNR vs Manual GT (Higher is better)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("PSNR (dB)", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # LPIPS Plot
    for ep, vals in zip(epochs, raw_points['lpips']):
        x_jittered = np.random.normal(loc=ep, scale=jitter_width, size=len(vals))
        axes[2].scatter(x_jittered, vals, color='#d62728', alpha=0.4, s=28, edgecolors='none')
    axes[2].plot(epochs, stats['lpips_mean'], marker='^', markersize=6, color='#7a1112', lw=2.5, label='Mean LPIPS')
    axes[2].scatter([], [], color='#d62728', alpha=0.5, s=28, label='Per-Image LPIPS')
    axes[2].set_title("Object LPIPS vs Manual GT (Lower is better)", fontsize=13, fontweight='bold')
    axes[2].set_ylabel("LPIPS Distance", fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    step = max(1, len(epochs) // 12)
    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_xticks(epochs[::step])
        ax.legend(loc='best', framealpha=0.85)

    plt.suptitle(f"Manual Foreground RGB Evaluation — {opt.name}", fontsize=15, fontweight='bold')
    plt.tight_layout()

    chart_path = output_dir / f"{opt.name}_removeback_rgb_metrics_panel.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved metric chart: {chart_path}")
    if opt.save_debug_images:
        print(f"Check debug images in: {debug_save_dir}")


if __name__ == "__main__":
    evaluate()