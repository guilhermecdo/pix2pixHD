from email import parser

import cv2
import os
import random
import argparse
import shutil
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Unified Sonar Data Pre-processing")
    parser.add_argument("--src", type=str, required=True, help="Path to raw CroppedImages folder")
    parser.add_argument("--dst", type=str, default="./datasets/Oculus2Didson", help="Path to pix2pixHD dataset folder")
    parser.add_argument("--mode", type=str, choices=['pad', 'interp'], required=True, help="Method: 'pad' or 'interp'")
    parser.add_argument("--size", type=int, default=256, help="Final image size (width and height). Must be multiple of 32.")
    parser.add_argument("--split", type=float, default=0.15, help="Fraction of data for testing (e.g., 0.15 for 15%%)")
    return parser.parse_args()

def process_image(img, mode, target_size):
    if mode == 'interp':
        # Lanczos4 is excellent for preserving acoustic intensity peaks
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    else: # Padding
        h, w = img.shape[:2]
        if h > target_size or w > target_size:
            print(f"Warning: Image ({h}x{w}) is larger than target size ({target_size}). Resizing down first.")
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            h, w = target_size, target_size

        top = (target_size - h) // 2
        bottom = target_size - h - top
        left = (target_size - w) // 2
        right = target_size - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def main():
    args = get_args()
    
    # Validation: pix2pixHD architecture requires dimensions divisible by 32
    if args.size % 32 != 0:
        print(f"Error: Size {args.size} is not divisible by 32. This will cause architecture mismatches in pix2pixHD.")
        return

    src = Path(args.src)
    dst = Path(args.dst)

    # Create directory structure
    for phase in ['train', 'test']:
        (dst / f"{phase}_A").mkdir(parents=True, exist_ok=True)
        (dst / f"{phase}_B").mkdir(parents=True, exist_ok=True)

    # Gather pairs
    all_files = os.listdir(src)
    oculus_files = [f for f in all_files if f.startswith('O') and f.endswith('.jpg')]
    
    valid_pairs = []
    for f in oculus_files:
        didson_name = 'D' + f[1:]
        if os.path.exists(src / didson_name):
            valid_pairs.append(f)

    # Random Split
    random.seed(42) # Fixed seed for reproducible splits
    random.shuffle(valid_pairs)
    num_test = int(len(valid_pairs) * args.split)
    test_pairs = valid_pairs[:num_test]
    train_pairs = valid_pairs[num_test:]

    print(f"Total pairs found: {len(valid_pairs)}")
    print(f"Mode: {args.mode.upper()} | Target Size: {args.size}x{args.size}")
    print(f"Splitting: {len(train_pairs)} for training, {len(test_pairs)} for testing.")

    for phase, pairs in [('train', train_pairs), ('test', test_pairs)]:
        print(f"Processing {phase} set...")
        for f in pairs:
            didson_name = 'D' + f[1:]
            common_name = f[1:] 

            img_a = cv2.imread(str(src / f))
            img_b = cv2.imread(str(src / didson_name))
            
            if img_a is None or img_b is None:
                continue

            proc_a = process_image(img_a, args.mode, args.size)
            proc_b = process_image(img_b, args.mode, args.size)

            cv2.imwrite(str(dst / f"{phase}_A" / common_name), proc_a)
            cv2.imwrite(str(dst / f"{phase}_B" / common_name), proc_b)

    print(f"\nDone! Dataset ready at {dst}")

if __name__ == "__main__":
    main()