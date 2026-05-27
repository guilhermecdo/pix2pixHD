#!/bin/bash

# Removed set -e to prevent silent crashes from minor string issues
# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
RAW_DATA_ROOT="/home/guilherme/git/pix2pixHD/datasets/Sonar2Optical"
STAGE_SUBSET="./datasets/s2o_subset_color"
STAGE_GRAYSCALE="./datasets/s2o_subset_grayscale"
IMG_SIZE=512

echo "========================================================="
echo "   Starting Sub-Sampling & Grayscale Translation Runs   "
echo "========================================================="

echo "Cleaning old staging directories..."
rm -rf "$STAGE_SUBSET" "$STAGE_GRAYSCALE"

for PHASE in "train" "test"; do
    src_A="${RAW_DATA_ROOT}/${PHASE}_A"
    src_B="${RAW_DATA_ROOT}/${PHASE}_B"
    
    mkdir -p "${STAGE_SUBSET}/${PHASE}_A" "${STAGE_SUBSET}/${PHASE}_B"
    mkdir -p "${STAGE_GRAYSCALE}/${PHASE}_A" "${STAGE_GRAYSCALE}/${PHASE}_B"

    if [ ! -d "$src_A" ] || [ ! -d "$src_B" ]; then 
        echo "Error: Missing directory for phase ${PHASE}"
        continue
    fi

    # Read files into arrays
    files_A=($(find "$src_A" -maxdepth 1 -type f \( -name "Sonar_*" \) | sort))
    files_B=($(find "$src_B" -maxdepth 1 -type f \( -name "RGB_*" \) | sort))
    
    count_A=${#files_A[@]}
    count_B=${#files_B[@]}
    
    echo "Processing ${PHASE} split: Found Sonar=$count_A, RGB=$count_B images."

    # Determine loop limit
    limit=$(( count_A < count_B ? count_A : count_B ))
    if [ "$limit" -eq 0 ]; then
        echo "Warning: Zero matching files found for phase ${PHASE}. Check file prefixes!"
        continue
    fi

    idx_counter=0
    for ((i=0; i<limit; i+=5)); do
        file_A="${files_A[$i]}"
        file_B="${files_B[$i]}"
        
        ext="${file_A##*.}"
        padded_idx=$(printf "%05d" $idx_counter)
        new_filename="frame_${padded_idx}.${ext}"

        abs_A="$(cd "$(dirname "$file_A")" && pwd)/$(basename "$file_A")"
        abs_B="$(cd "$(dirname "$file_B")" && pwd)/$(basename "$file_B")"

        # --- CONFIG 1: 1/5 Color Subset ---
        ln -sf "$abs_A" "${STAGE_SUBSET}/${PHASE}_A/${new_filename}"
        ln -sf "$abs_B" "${STAGE_SUBSET}/${PHASE}_B/${new_filename}"

        # --- CONFIG 2: 1/5 Grayscale Subset ---
        ln -sf "$abs_A" "${STAGE_GRAYSCALE}/${PHASE}_A/${new_filename}"
        
        python3 -c "
import cv2
img = cv2.imread('$abs_B')
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.merge([gray, gray, gray])
    cv2.imwrite('${STAGE_GRAYSCALE}/${PHASE}_B/${new_filename}', gray_3ch)
"
        ((idx_counter++))
    done
    echo "Successfully generated $idx_counter matched subset pairs for ${PHASE} split."
done

# ==========================================
# 2. RUN EXPERIMENT 1: 1/5 COLOR SUBSET
# ==========================================
EXP_1="s2o_1fifth_color"
echo "---------------------------------------------------------"
echo ">>> [EXPERIMENT 1/2] Training on 1/5 Color Data Split..."
echo "---------------------------------------------------------"

# python3 train.py --name "$EXP_1" --dataroot "$STAGE_SUBSET" \
#   --no_instance --label_nc 0 \
#   --resize_or_crop resize_and_crop --loadSize $IMG_SIZE --fineSize $IMG_SIZE \
#   --nThreads 0 --gpu_ids 0 --no_vgg_loss --batchSize 6 \
#   --niter 100 --niter_decay 100 --save_epoch_freq 50

echo ">>> Inference & Reporting for Experiment 1..."
python3 test.py --name "$EXP_1" --dataroot "$STAGE_SUBSET" \
  --no_instance --label_nc 0 \
  --resize_or_crop resize_and_crop --loadSize $IMG_SIZE --fineSize $IMG_SIZE \
  --nThreads 0 --gpu_ids 0 --how_many 500

python3 build_final_report.py --results_dir "./results/${EXP_1}/test_latest/images" --dataset_dir "$STAGE_SUBSET"
python3 create_results_video.py --results_dir "./results/${EXP_1}/test_latest/images" --output_video "./results/${EXP_1}_summary.mp4"

# ==========================================
# 3. RUN EXPERIMENT 2: 1/5 GRAYSCALE SUBSET
# ==========================================
# EXP_2="s2o_1fifth_grayscale"
# echo "---------------------------------------------------------"
# echo ">>> [EXPERIMENT 2/2] Training on 1/5 Grayscale Data Split..."
# echo "---------------------------------------------------------"

# python3 train.py --name "$EXP_2" --dataroot "$STAGE_GRAYSCALE" \
#   --no_instance --label_nc 0 \
#   --resize_or_crop resize_and_crop --loadSize $IMG_SIZE --fineSize $IMG_SIZE \
#   --nThreads 0 --gpu_ids 0 --no_vgg_loss --batchSize 1 \
#   --niter 100 --niter_decay 100 --save_epoch_freq 5

# echo ">>> Inference & Reporting for Experiment 2..."
# python3 test.py --name "$EXP_2" --dataroot "$STAGE_GRAYSCALE" \
#   --no_instance --label_nc 0 \
#   --resize_or_crop resize_and_crop --loadSize $IMG_SIZE --fineSize $IMG_SIZE \
#   --nThreads 0 --gpu_ids 0 --how_many 500

# python3 build_final_report.py --results_dir "./results/${EXP_2}/test_latest/images" --dataset_dir "$STAGE_GRAYSCALE"
# python3 create_results_video.py --results_dir "./results/${EXP_2}/test_latest/images" --output_video "./results/${EXP_2}_summary.mp4"

# echo "========================================================="
# echo " All experiments processed successfully! "
# echo "========================================================="