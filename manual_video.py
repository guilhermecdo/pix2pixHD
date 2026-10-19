import os
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


def get_args():
    parser = argparse.ArgumentParser(
        description="Create side-by-side video mixing good and bad result folders in random order with original captions and running bad image counter."
    )
    parser.add_argument("--good_dir", type=str, required=True,
                        help="Path to folder containing good inference results (*_synthesized Image, *_Input Label, *_Real Image)")
    parser.add_argument("--bad_dir", type=str, required=True,
                        help="Path to folder containing bad inference results")
    parser.add_argument("--output_video", type=str, default="translation_results.mp4",
                        help="Path of the generated MP4 video")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second for output video (default: 1)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Height and width for each individual panel image (default: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for repeatable shuffling order (default: 42)")
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


def collect_triplets(folder_path, is_bad):
    """
    Finds synthesized images and resolves corresponding Input Label and Real Image files.
    Returns list of dicts: [{'gen': path, 'sonar': path, 'gt': path, 'is_bad': bool, 'prefix': str}, ...]
    """
    base = Path(folder_path).resolve()
    gen_files = sorted(list(base.glob("*_synthesized Image.jpg"))) + sorted(list(base.glob("*_synthesized Image.png")))

    triplets = []
    for gen_file in gen_files:
        suffix = "_synthesized Image" + gen_file.suffix
        prefix = gen_file.name.replace(suffix, "")

        sonar_file = base / f"{prefix}_Input Label{gen_file.suffix}"
        if not sonar_file.exists():
            sonar_file = base / f"{prefix}_input_label{gen_file.suffix}"

        true_opt_file = base / f"{prefix}_Real Image{gen_file.suffix}"
        if not true_opt_file.exists():
            true_opt_file = base / f"{prefix}_real_image{gen_file.suffix}"

        if sonar_file.exists() and true_opt_file.exists():
            triplets.append({
                "gen": gen_file,
                "sonar": sonar_file,
                "gt": true_opt_file,
                "is_bad": is_bad,
                "prefix": prefix
            })

    return triplets


def make_video(good_dir, bad_dir, output_video, fps, img_size, seed, max_images, no_rembg):
    # Set random seed for repeatable ordering
    random.seed(seed)

    print(f"Scanning 'Good' directory: {good_dir}")
    good_triplets = collect_triplets(good_dir, is_bad=False)
    print(f"Found {len(good_triplets)} valid triplets in Good folder.")

    print(f"Scanning 'Bad' directory:  {bad_dir}")
    bad_triplets = collect_triplets(bad_dir, is_bad=True)
    print(f"Found {len(bad_triplets)} valid triplets in Bad folder.")

    all_triplets = good_triplets + bad_triplets
    if not all_triplets:
        print("Error: No valid frame triplets found across either directory.")
        return

    # Shuffle both pools together in random order
    random.shuffle(all_triplets)

    if max_images is not None and max_images > 0:
        all_triplets = all_triplets[:max_images]

    total_eval_cap = len(all_triplets)
    use_rembg = not no_rembg
    rembg_status_str = "Disabled (Full Frame)" if no_rembg else "Enabled (Foreground Only)"

    print(f"Found {total_eval_cap} frames to process.")
    print(f"Background Removal: {rembg_status_str}")

    rembg_session = None
    if use_rembg:
        from rembg import new_session
        rembg_session = new_session("u2net")

    video_width = img_size * 3
    video_height = img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_path = Path(output_video).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (video_width, video_height))

    count = 0
    bad_count = 0

    for item in all_triplets:
        img_sonar = cv2.imread(str(item["sonar"]))
        img_true_opt = cv2.imread(str(item["gt"]))
        img_gen_opt = cv2.imread(str(item["gen"]))

        if img_sonar is None or img_true_opt is None or img_gen_opt is None:
            continue

        img_sonar_resized = cv2.resize(img_sonar, (img_size, img_size))
        img_true_opt_resized = cv2.resize(img_true_opt, (img_size, img_size))
        img_gen_opt_resized = cv2.resize(img_gen_opt, (img_size, img_size))

        gray_gt = cv2.cvtColor(img_true_opt_resized, cv2.COLOR_BGR2GRAY)
        gray_gen = cv2.cvtColor(img_gen_opt_resized, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # SSIM Evaluation: Foreground masked vs. Full-frame
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
        is_bad = item["is_bad"]
        if is_bad:
            bad_count += 1

        # -------------------------------------------------------------
        # Annotate Video Frame Panels (Matching Original Style)
        # -------------------------------------------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_sonar_resized, "Sonar Input", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_true_opt_resized, "True Optical (GT)", (20, 35), font, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        status_color = (0, 0, 255) if is_bad else (0, 255, 0)
        gen_label = f"Generated (SSIM: {frame_ssim:.2f}) | Bad Frames: {bad_count}/{total_eval_cap}"
        cv2.putText(img_gen_opt_resized, gen_label, (20, 35), font, 0.55, status_color, 2, cv2.LINE_AA)

        combined_frame = np.hstack((img_sonar_resized, img_true_opt_resized, img_gen_opt_resized))
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"\nDone! Video saved to: {output_video}")
    print(f"Total Video Frames Processed: {count}")
    print(f"Total Bad Frames: {bad_count}/{count}\n")


if __name__ == "__main__":
    args = get_args()
    make_video(
        good_dir=args.good_dir,
        bad_dir=args.bad_dir,
        output_video=args.output_video,
        fps=args.fps,
        img_size=args.img_size,
        seed=args.seed,
        max_images=args.max_images,
        no_rembg=args.no_rembg
    )