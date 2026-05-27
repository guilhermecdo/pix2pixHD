import cv2
import os
import argparse
from pathlib import Path
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Create side-by-side result video.")
    parser.add_argument("--results_dir", type=str, required=True, 
                        help="Path to the test_latest/images folder containing results")
    parser.add_argument("--output_video", type=str, default="translation_results.mp4", 
                        help="Name of the output video file")
    parser.add_argument("--fps", type=int, default=1, 
                        help="Frames per second. Default 1 means each image pair stays for 1 second.")
    parser.add_argument("--img_size", type=int, default=512, 
                        help="Height and width for each individual panel image")
    return parser.parse_args()

def make_video(results_dir, output_video, fps, img_size):
    res_path = Path(results_dir).resolve()
    
    # 1. Find all synthesized images to use as the base indexing loop
    synthesized_files = sorted(list(res_path.glob("*_synthesized_image.jpg")))
    
    if not synthesized_files:
        print(f"Error: No '_synthesized_image.jpg' files found in {results_dir}")
        print("Make sure you run test.py and build your report scripts first!")
        return

    print(f"Found {len(synthesized_files)} result sets. Preparing video layout...")

    # Calculate dimensions for 3 images side-by-side
    # Width = 3 * img_size, Height = img_size
    video_width = img_size * 3
    video_height = img_size

    # 2. Initialize Video Writer
    # Using 'mp4v' or 'XVID' codec for reliable MP4 generation across Linux/Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (video_width, video_height))

    count = 0
    for gen_file in synthesized_files:
        # Extract the base frame identifier index (e.g., frame_00001)
        prefix = gen_file.name.replace("_synthesized_image.jpg", "")
        
        # Define paths for your specific ordered sequence matching your new test bindings
        sonar_file = res_path / f"{prefix}_src_oculus.jpg"
        true_opt_file = res_path / f"{prefix}_gt_didson.jpg"  # The real target array
        
        # Verify all three elements exist before constructing the video block frame
        if not sonar_file.exists() or not true_opt_file.exists  ():
            # Fallback check if the script used old naming configurations
            sonar_file = res_path / f"{prefix}_report_input.jpg"
            true_opt_file = res_path / f"{prefix}_report_gt.jpg"
            
            if not sonar_file.exists():
                print(f"Skipping missing sequence pair for layout ID: {prefix}")
                continue

        # 3. Load and uniformly shape images
        img_sonar = cv2.imread(str(sonar_file))
        img_true_opt = cv2.imread(str(true_opt_file))
        img_gen_opt = cv2.imread(str(gen_file))

        # Check for corrupted reads
        if img_sonar is None or img_true_opt is None or img_gen_opt is None:
            continue

        # Resize each panel to be exactly square so they match perfectly
        img_sonar = cv2.resize(img_sonar, (img_size, img_size))
        img_true_opt = cv2.resize(img_true_opt, (img_size, img_size))
        img_gen_opt = cv2.resize(img_gen_opt, (img_size, img_size))

        # Optional: Add clear overlay text label headers onto the panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_sonar, "Sonar Input", (20, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_true_opt, "True Optical (GT)", (20, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img_gen_opt, "Generated Optical", (20, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 4. Stitch horizontally: [ Sonar | True Optical | Generated Optical ]
        combined_frame = np.hstack((img_sonar, img_true_opt, img_gen_opt))

        # Write frame to video sequence
        video_writer.write(combined_frame)
        count += 1

    video_writer.release()
    print(f"\nSuccess! Video saved to: {output_video}")
    print(f"Total Video Duration: {count} seconds at {fps} frame(s) per second.")

if __name__ == "__main__":
    args = get_args()
    make_video(args.results_dir, args.output_video, args.fps, args.img_size)