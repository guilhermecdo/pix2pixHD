import os
import re
import cv2
import csv
import shutil
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# pix2pixHD native imports
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util


def get_args():
    test_opt = TestOptions()
    test_opt.parser.add_argument("--ssim_threshold", type=float, default=0.70,
                                 help="SSIM cutoff to separate images (default: 0.70)")
    test_opt.parser.add_argument("--output_partition_dir", type=str, default="./dataset_ssim_partitioned",
                                 help="Folder where below/above threshold sets will be stored")
    test_opt.parser.add_argument("--save_synthesized", action="store_true",
                                 help="Also copy synthesized fake images into the partitioned folders")
    test_opt.parser.add_argument("--action", type=str, default="copy", choices=["copy", "move"],
                                 help="Whether to 'copy' or 'move' paired files into threshold directories")
    opt = test_opt.parse(save=False)
    return opt


def run_phase_evaluation(opt, model, phase_name, output_base, csv_writer):
    # Enforce strict deterministic evaluation transforms for both train and test phases
    opt.phase = phase_name
    opt.isTrain = False
    opt.no_flip = True
    opt.serial_batches = True
    opt.how_many = float("inf")

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    print(f"\nProcessing phase: [{phase_name}] ({len(dataset)} samples)...")

    below_dir = output_base / "below_threshold"
    above_dir = output_base / "above_threshold"

    below_count = 0
    above_count = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc=f"Evaluating {phase_name}")):
            # Resolve file paths explicitly first
            path_a = data['path'][0] if 'path' in data else None
            path_b = data['B_paths'][0] if 'B_paths' in data else None

            # Fallback path lookup if B_paths is not present
            if path_a and not path_b:
                stem_a = Path(path_a).stem
                parent_b = Path(path_a).parent.parent / f"{phase_name}_B"
                num_matches = re.findall(r"\d+", stem_a)
                if num_matches:
                    match_id = num_matches[-1]
                    matches = list(parent_b.glob(f"*{match_id}*"))
                else:
                    matches = []
                path_b = str(matches[0]) if matches else None

            # 1. Forward Pass: In pix2pixHD with label_nc=0, input Sonar comes from data['label']
            # If data['label'] is empty, it uses data['image']
            if opt.label_nc == 0:
                if 'label' in data and data['label'].nelement() > 0:
                    input_tensor = data['label']
                else:
                    input_tensor = data['image']
                generated = model.inference(input_tensor, data['inst'])
            else:
                generated = model.inference(data['label'], data['inst'], data['image'])

            fake_rgb = util.tensor2im(generated.data[0])

            # 2. Ground Truth: Read the actual GT Optical file directly from disk to guarantee ground-truth validity
            if path_b and os.path.exists(path_b):
                real_bgr = cv2.imread(str(path_b))
                real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB)
            else:
                real_rgb = util.tensor2im(data['image'][0])

            # Resize if dimensions differ
            if fake_rgb.shape[:2] != real_rgb.shape[:2]:
                fake_rgb = cv2.resize(fake_rgb, (real_rgb.shape[1], real_rgb.shape[0]))

            # 3. Calculate SSIM across full RGB
            score = ssim(fake_rgb, real_rgb, channel_axis=2, data_range=255)

            # Debug the first 3 images to stdout
            if i < 3:
                print(f"\n[DEBUG Frame {i}] A: {Path(path_a).name if path_a else 'None'} | B: {Path(path_b).name if path_b else 'None'} | SSIM: {score:.4f}")

            is_above = score >= opt.ssim_threshold
            target_subfolder = above_dir if is_above else below_dir
            if is_above:
                above_count += 1
            else:
                below_count += 1

            target_a_dir = target_subfolder / f"{phase_name}_A"
            target_b_dir = target_subfolder / f"{phase_name}_B"
            target_a_dir.mkdir(parents=True, exist_ok=True)
            target_b_dir.mkdir(parents=True, exist_ok=True)

            if path_a and os.path.exists(path_a):
                dest_a = target_a_dir / Path(path_a).name
                if opt.action == "move":
                    shutil.move(path_a, str(dest_a))
                else:
                    shutil.copy2(path_a, str(dest_a))

            if path_b and os.path.exists(path_b):
                dest_b = target_b_dir / Path(path_b).name
                if opt.action == "move":
                    shutil.move(path_b, str(dest_b))
                else:
                    shutil.copy2(path_b, str(dest_b))

            if opt.save_synthesized:
                target_synth_dir = target_subfolder / f"{phase_name}_synth"
                target_synth_dir.mkdir(parents=True, exist_ok=True)
                synth_name = Path(path_b).stem if path_b else f"sample_{i:05d}"
                bgr_synth = cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(target_synth_dir / f"{synth_name}_synth.png"), bgr_synth)

            csv_writer.writerow([
                phase_name,
                Path(path_a).name if path_a else f"sample_{i}_A",
                Path(path_b).name if path_b else f"sample_{i}_B",
                f"{score:.6f}",
                "above" if is_above else "below"
            ])

    return below_count, above_count


def main():
    opt = get_args()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    output_base = Path(opt.output_partition_dir).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = create_model(opt)
    if hasattr(model, 'netG'):
        model.netG.eval()
    elif hasattr(model, 'module') and hasattr(model.module, 'netG'):
        model.module.netG.eval()

    csv_path = output_base / f"{opt.name}_ssim_split_log.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Split_Phase", "Image_A", "Image_B", "SSIM", "Category"])

        total_below = 0
        total_above = 0

        # Run on train set if train folders exist
        train_a_path = Path(opt.dataroot) / "train_A"
        if train_a_path.exists():
            b_cnt, a_cnt = run_phase_evaluation(opt, model, "train", output_base, writer)
            total_below += b_cnt
            total_above += a_cnt

        # Run on test set if test folders exist
        test_a_path = Path(opt.dataroot) / "test_A"
        if test_a_path.exists():
            b_cnt, a_cnt = run_phase_evaluation(opt, model, "test", output_base, writer)
            total_below += b_cnt
            total_above += a_cnt

    print("\n=======================================================")
    print(f"Partitioning Completed across all data!")
    print(f"SSIM Threshold: {opt.ssim_threshold:.2f}")
    print(f"Total Below Threshold: {total_below}")
    print(f"Total Above Threshold: {total_above}")
    print(f"Destination: {output_base}")
    print(f"CSV Log:     {csv_path}")
    print("=======================================================")


if __name__ == "__main__":
    main()