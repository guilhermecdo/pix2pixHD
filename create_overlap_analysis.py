import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

def get_args():
    parser = argparse.ArgumentParser(description="Create Red/Green superposition diagnostic video and split saved frames into above/below threshold folders.")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Path to the test_latest/images folder containing generated results")
    parser.add_argument("--output_video", type=str, default="overlap_analysis.mp4", 
                        help="Path/name for the output video file")
    parser.add_argument("--output_frames_dir", type=str, default="./ablation_overlap_results",
                        help="Base directory to save individual superposition image frames")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for subfolder organization")
    parser.add_argument("--fps", type=int, default=1, 
                        help="Video framerate (default: 1 fps)")
    parser.add_argument("--img_size", type=int, default=512, 
                        help="Resolution (HxW) per panel")
    parser.add_argument("--max_images", "-n", type=int, default=None, 
                        help="Maximum frames to process")
    parser.add_argument("--ssim_threshold", type=float, default=0.75,
                        help="SSIM threshold used to divide results into below_threshold and above_threshold (default: 0.75)")
    parser.add_argument("--use_rembg", action="store_true",
                        help="Enable rembg background removal; saves results in <exp_name>_removebackground")
    return parser.parse_args()

def to_grayscale(img_bgr):
    """Convert an image to a single-channel 2D uint8 array."""
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr.copy()

def get_foreground_mask(session, img_bgr):
    """Extract binary foreground mask using rembg from a BGR image."""
    from rembg import remove
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(img_rgb, session=session)
    alpha = rgba[:, :, 3]
    return (alpha > 10).astype(np.uint8)

def make_overlap_video():
    args = get_args()
    res_path = Path(args.results_dir).resolve()

    # Find synthesized frames
    synthesized_files = sorted(list(res_path.glob("*_synthesized Image.jpg")))
    if not synthesized_files:
        synthesized_files = sorted(list(res_path.glob("*_synthesized Image.png")))

    if not synthesized_files:
        print(f"Error: No '_synthesized Image' files found in {res_path}")
        return

    exp_name = args.exp_name if args.exp_name is not None else res_path.parent.parent.name
    if args.max_images is not None and args.max_images > 0:
        synthesized_files = synthesized_files[:args.max_images]

    # Append '_removebackground' to experiment directory name if rembg is requested
    folder_exp_name = f"{exp_name}_removebackground" if args.use_rembg else exp_name
    base_out_path = Path(args.output_frames_dir).resolve() / folder_exp_name
    
    # Create distinct directories for below and above the threshold
    below_thresh_dir = base_out_path / "below_threshold"
    above_thresh_dir = base_out_path / "above_threshold"
    below_thresh_dir.mkdir(parents=True, exist_ok=True)
    above_thresh_dir.mkdir(parents=True, exist_ok=True)

    rembg_session = None
    if args.use_rembg:
        from rembg import new_session
        print("Initializing rembg session ('u2net')...")
        rembg_session = new_session("u2net")

    print(f"[{exp_name}] Processing {len(synthesized_files)} frames for Red/Green overlap analysis.")
    print(f"Background Removal: {'ENABLED' if args.use_rembg else 'DISABLED'}")
    print(f"SSIM Threshold: {args.ssim_threshold:.2f}")
    print(f"Saving outputs to:")
    print(f"  - Below (< {args.ssim_threshold:.2f}): {below_thresh_dir}")
    print(f"  - Above (>= {args.ssim_threshold:.2f}): {above_thresh_dir}")

    panel_w = args.img_size
    video_width = panel_w * 4
    video_height = args.img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (video_width, video_height))

    count = 0
    below_count = 0
    above_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for gen_file in synthesized_files:
        suffix = "_synthesized Image" + gen_file.suffix
        prefix = gen_file.name.replace(suffix, "")

        sonar_file = res_path / f"{prefix}_Input Label{gen_file.suffix}"
        true_opt_file = res_path / f"{prefix}_Real Image{gen_file.suffix}"

        if not sonar_file.exists():
            sonar_file = res_path / f"{prefix}_input_label{gen_file.suffix}"
        if not true_opt_file.exists():
            true_opt_file = res_path / f"{prefix}_real_image{gen_file.suffix}"

        if not sonar_file.exists() or not true_opt_file.exists():
            continue

        img_sonar = cv2.imread(str(sonar_file))
        img_true_opt = cv2.imread(str(true_opt_file))
        img_gen_opt = cv2.imread(str(gen_file))

        if img_sonar is None or img_true_opt is None or img_gen_opt is None:
            continue

        sonar_resized = cv2.resize(img_sonar, (panel_w, args.img_size))
        gt_resized = cv2.resize(img_true_opt, (panel_w, args.img_size))
        gen_resized = cv2.resize(img_gen_opt, (panel_w, args.img_size))

        gt_gray = to_grayscale(gt_resized)
        gen_gray = to_grayscale(gen_resized)

        # -------------------------------------------------------------
        # Optional rembg: Mask background to zero
        # -------------------------------------------------------------
        if args.use_rembg:
            mask = get_foreground_mask(rembg_session, gt_resized)
            gt_gray = cv2.bitwise_and(gt_gray, gt_gray, mask=mask)
            gen_gray = cv2.bitwise_and(gen_gray, gen_gray, mask=mask)

            # Bounding box for tight SSIM
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                ymin, ymax = np.min(y_indices), np.max(y_indices)
                xmin, xmax = np.min(x_indices), np.max(x_indices)
                crop_gt = gt_gray[ymin:ymax+1, xmin:xmax+1]
                crop_gen = gen_gray[ymin:ymax+1, xmin:xmax+1]
            else:
                crop_gt, crop_gen = gt_gray, gen_gray

            if crop_gt.shape[0] < 7 or crop_gt.shape[1] < 7:
                frame_ssim = ssim(gen_gray, gt_gray, data_range=255)
            else:
                frame_ssim = ssim(crop_gen, crop_gt, data_range=255)
        else:
            frame_ssim = ssim(gt_gray, gen_gray, data_range=255)

        # -------------------------------------------------------------
        # Color Scale Projections
        # -------------------------------------------------------------
        # GT in Pure Green: B=0, G=gt_gray, R=0
        green_gt = np.zeros((args.img_size, panel_w, 3), dtype=np.uint8)
        green_gt[:, :, 1] = gt_gray

        # Generated in Pure Red: B=0, G=0, R=gen_gray
        red_gen = np.zeros((args.img_size, panel_w, 3), dtype=np.uint8)
        red_gen[:, :, 2] = gen_gray

        # Superposition / Sobreposição (Yellow = Match)
        overlap = np.zeros((args.img_size, panel_w, 3), dtype=np.uint8)
        overlap[:, :, 1] = gt_gray    # Green channel = GT
        overlap[:, :, 2] = gen_gray   # Red channel = Generated

        # -------------------------------------------------------------
        # Save Frames into Respective Category Directory
        # -------------------------------------------------------------
        if frame_ssim < args.ssim_threshold:
            target_save_dir = below_thresh_dir
            below_count += 1
        else:
            target_save_dir = above_thresh_dir
            above_count += 1

        # 1. Save isolated images
        cv2.imwrite(str(target_save_dir / f"{prefix}_gt.png"), green_gt)
        cv2.imwrite(str(target_save_dir / f"{prefix}_gen.png"), red_gen)
        cv2.imwrite(str(target_save_dir / f"{prefix}_ssim_{frame_ssim:.2f}_overlap.png"), overlap)

        # 2. Save annotated side-by-side diagnostic strip
        diag_strip = np.hstack((sonar_resized, green_gt, red_gen, overlap))
        lbl = "Foreground" if args.use_rembg else "Full"
        cv2.putText(diag_strip, f"{lbl} SSIM: {frame_ssim:.2f} | Yellow=Match, Green=GT Only, Red=Gen Only", 
                    (20, args.img_size - 20), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(target_save_dir / f"{prefix}_ssim_{frame_ssim:.2f}_composite.png"), diag_strip)

        # -------------------------------------------------------------
        # Annotations for Video Stream
        # -------------------------------------------------------------
        cv2.putText(sonar_resized, "Sonar Input", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(green_gt, "GT (Green)", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(red_gen, "Gen (Red)", (20, 35), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        
        status_color = (0, 255, 0) if frame_ssim >= args.ssim_threshold else (0, 0, 255)
        cv2.putText(overlap, f"Overlap (SSIM: {frame_ssim:.2f})", (15, 35), font, 0.55, status_color, 2, cv2.LINE_AA)

        combined_frame = np.hstack((sonar_resized, green_gt, red_gen, overlap))
        writer.write(combined_frame)
        count += 1

    writer.release()
    print(f"\n[{exp_name}] Finished! Video -> {args.output_video}")
    print(f"Total Frames Processed: {count}")
    print(f"  - Saved to below_threshold (< {args.ssim_threshold:.2f}): {below_count}")
    print(f"  - Saved to above_threshold (>= {args.ssim_threshold:.2f}): {above_count}")
    print(f"Root Folder: {base_out_path}\n")

if __name__ == "__main__":
    make_overlap_video()