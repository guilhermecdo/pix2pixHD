import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

def get_args():
    parser = argparse.ArgumentParser(description="Create side-by-side result video with optional rembg, low SSIM counter, and triplet logger.")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Path to the test_latest/images folder containing results")
    parser.add_argument("--output_video", type=str, default="translation_results.mp4", 
                        help="Name of the output video file")
    parser.add_argument("--fps", type=int, default=1, 
                        help="Frames per second.")
    parser.add_argument("--img_size", type=int, default=512, 
                        help="Height and width for each individual panel image")
    parser.add_argument("--ssim_threshold", type=float, default=0.75,
                        help="SSIM cutoff threshold below which frames are saved (default: 0.75)")
    parser.add_argument("--bad_results_dir", type=str, default="./bad_results",
                        help="Base directory to store low SSIM triplets")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name used to create subfolder under bad_results_dir")
    parser.add_argument("--max_images", "-n", type=int, default=None,
                        help="Maximum number of frames/triplets to process for the video")
    parser.add_argument("--no_rembg", action="store_true",
                        help="Disable background removal with rembg and evaluate full image instead")
    return parser.parse_args()

def get_foreground_mask(session, img_bgr):
    """Extract binary foreground mask using rembg from a BGR image."""
    from rembg import remove
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(img_rgb, session=session)
    alpha = rgba[:, :, 3]
    return (alpha > 10).astype(np.uint8)

def make_video(results_dir, output_video, fps, img_size, ssim_threshold, bad_results_dir, exp_name, max_images, no_rembg):
    res_path = Path(results_dir).resolve()
    
    synthesized_files = sorted(list(res_path.glob("*_synthesized Image.jpg")))
    if not synthesized_files:
        synthesized_files = sorted(list(res_path.glob("*_synthesized Image.png")))
    
    if not synthesized_files:
        print(f"Error: No '_synthesized Image' files found in {results_dir}")
        return

    if exp_name is None:
        exp_name = res_path.parent.parent.name

    if max_images is not None and max_images > 0:
        synthesized_files = synthesized_files[:max_images]

    bad_dir = Path(bad_results_dir).resolve() / exp_name
    bad_dir.mkdir(parents=True, exist_ok=True)

    use_rembg = not no_rembg
    rembg_status_str = "Disabled (Full Frame)" if no_rembg else "Enabled (Foreground Only)"
    print(f"Found {len(synthesized_files)} frames to process for '{exp_name}'.")
    print(f"Background Removal: {rembg_status_str}")
    print(f"Tracking frames with SSIM < {ssim_threshold:.2f}...")

    rembg_session = None
    if use_rembg:
        from rembg import new_session
        rembg_session = new_session("u2net")

    video_width = img_size * 3
    video_height = img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))

    count = 0
    bad_count = 0
    metric_name = "SSIM_Full" if no_rembg else "SSIM_Foreground"
    log_lines = [f"Prefix\t{metric_name}\n"]

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

        img_sonar_resized = cv2.resize(img_sonar, (img_size, img_size))
        img_true_opt_resized = cv2.resize(img_true_opt, (img_size, img_size))
        img_gen_opt_resized = cv2.resize(img_gen_opt, (img_size, img_size))

        gray_gt = cv2.cvtColor(img_true_opt_resized, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(img_gen_opt_resized, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # Conditional rembg mask application vs. full frame
        # -------------------------------------------------------------
        if use_rembg:
            mask = get_foreground_mask(rembg_session, img_true_opt_resized)
            real_masked = cv2.bitwise_and(gray_gt, gray_gt, mask=mask)
            fake_masked = cv2.bitwise_and(gray_gen, gray_gen, mask=mask)

            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                ymin, ymax = np.min(y_indices), np.max(y_indices)
                xmin, xmax = np.min(x_indices), np.max(x_indices)
                crop_fake = fake_masked[ymin:ymax+1, xmin:xmax+1]
                crop_real = real_masked[ymin:ymax+1, xmin:xmax+1]
            else:
                crop_fake = fake_masked
                crop_real = real_masked

            if crop_fake.shape[0] < 7 or crop_fake.shape[1] < 7:
                frame_ssim = ssim(fake_masked, real_masked, data_range=255)
            else:
                frame_ssim = ssim(crop_fake, crop_real, data_range=255)
        else:
            frame_ssim = ssim(gray_gen, gray_gt, data_range=255)

        count += 1
        is_bad = frame_ssim < ssim_threshold
        
        if is_bad:
            bad_count += 1
            cv2.imwrite(str(bad_dir / f"{prefix}_ssim_{frame_ssim:.2f}_input.jpg"), img_sonar)
            cv2.imwrite(str(bad_dir / f"{prefix}_ssim_{frame_ssim:.2f}_gt.jpg"), img_true_opt)
            cv2.imwrite(str(bad_dir / f"{prefix}_ssim_{frame_ssim:.2f}_generated.jpg"), img_gen_opt)
            
            annotated_triplet = np.hstack((img_sonar_resized.copy(), img_true_opt_resized.copy(), img_gen_opt_resized.copy()))
            tag_text = f"{'Foreground' if use_rembg else 'Full'} SSIM: {frame_ssim:.2f}"
            cv2.putText(annotated_triplet, tag_text, (20, img_size - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(bad_dir / f"{prefix}_ssim_{frame_ssim:.2f}_stitched.jpg"), annotated_triplet)
            
            log_lines.append(f"{prefix}\t{frame_ssim:.6f}\n")

        # -------------------------------------------------------------
        # Annotate Video Frame Panels
        # -------------------------------------------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_sonar_resized, "Sonar Input", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_true_opt_resized, "True Optical (GT)", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        
        status_color = (0, 0, 255) if is_bad else (0, 255, 0)
        total_eval_cap = max_images if (max_images is not None and max_images > 0) else len(synthesized_files)
        gen_label = f"Generated (SSIM: {frame_ssim:.2f}) | Below Thresh (< {ssim_threshold:.2f}): {bad_count}/{total_eval_cap}"
        cv2.putText(img_gen_opt_resized, gen_label, (20, 35), font, 0.55, status_color, 2, cv2.LINE_AA)

        combined_frame = np.hstack((img_sonar_resized, img_true_opt_resized, img_gen_opt_resized))
        video_writer.write(combined_frame)

    video_writer.release()

    with open(bad_dir / "bad_results_log.txt", "w") as f:
        f.writelines(log_lines)

    print(f"\nDone: {exp_name}")
    print(f"Total Video Frames Processed: {count} -> {output_video}")
    print(f"Frames with SSIM < {ssim_threshold:.2f}: {bad_count}/{count}")
    print(f"Low SSIM triplets saved in: {bad_dir}\n")

if __name__ == "__main__":
    args = get_args()
    make_video(
        results_dir=args.results_dir,
        output_video=args.output_video,
        fps=args.fps,
        img_size=args.img_size,
        ssim_threshold=args.ssim_threshold,
        bad_results_dir=args.bad_results_dir,
        exp_name=args.exp_name,
        max_images=args.max_images,
        no_rembg=args.no_rembg
    )