#!/bin/bash
set -e

# ==========================================================
# Batch Video & Low-SSIM Triplet Extraction Runner
# ==========================================================

RESULTS_BASE="/home/guilherme/git/pix2pixHD/results"
SHARED_BAD_DIR="/home/guilherme/git/pix2pixHD/all_bad_results"

SSIM_THRESHOLD=0.70
FPS=1
IMG_SIZE=720

# Define target experiments and evaluation split phase: "EXPERIMENT_NAME|PHASE"
EXPERIMENTS=(
    "didson_cleaned_G_4|test_latest"
    #"didson_cleaned_2|test_latest"
    #"sonar2optical_didson_stretch|test_latest"
    #"sonar2optical_loop_3|test_latest|Experiment: Sonar to Optical (Loop 3 No Transfer Learning)"
    #"sonar2optical_loop_3_no_transfer|test_latest|Experiment: Sonar to Optical (Loop 3 Transfer Learning)"
    #"sonar2optical_loop_1+2|test_latest|Experiment: Sonar to Optical (Loop 1+2 Baseline)"
)

mkdir -p "$SHARED_BAD_DIR"

echo "========================================================="
echo " Starting Batch Video Generation with Header & rembg "
echo " Threshold: Foreground SSIM < ${SSIM_THRESHOLD}"
echo " Central Bad Results Dir: ${SHARED_BAD_DIR}"
echo "========================================================="

for EXP_ENTRY in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r EXP_NAME PHASE MAX_N CAPTION <<< "$EXP_ENTRY"
    
    IMG_DIR="${RESULTS_BASE}/${EXP_NAME}/${PHASE}/images"
    OUT_VID="${RESULTS_BASE}/${EXP_NAME}/${EXP_NAME}_summary.mp4"

    echo "---------------------------------------------------------"
    echo "Processing Experiment: ${EXP_NAME}"
    echo "Caption:               ${CAPTION}"
    echo "Limit:                 ${MAX_N} images"
    echo "Images Directory:      ${IMG_DIR}"
    echo "---------------------------------------------------------"

    if [ ! -d "$IMG_DIR" ]; then
        echo "Warning: Directory '${IMG_DIR}' not found. Skipping."
        continue
    fi

    # Build optional max_images argument
    N_ARG="26"
    if [ -n "$MAX_N" ] && [ "$MAX_N" -gt 0 ]; then
        N_ARG="--max_images $MAX_N"
    fi

    python3 create_videos_SSIM.py \
        --results_dir "$IMG_DIR" \
        --output_video "$OUT_VID" \
        --fps "$FPS" \
        --img_size "$IMG_SIZE" \
        --ssim_threshold "$SSIM_THRESHOLD" \
        --bad_results_dir "$SHARED_BAD_DIR" \
        --exp_name "$EXP_NAME" \
        --max_images "$N_ARG" \
        --no_rembg

done

echo "========================================================="
echo " All batch runs finished! "
echo " Check individual experiment folders in: ${SHARED_BAD_DIR}"
echo "========================================================="