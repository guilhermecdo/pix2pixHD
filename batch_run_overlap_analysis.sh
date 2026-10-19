#!/bin/bash
set -e

# ==========================================================
# Batch Red/Green Superposition & Ablation Analysis Runner
# ==========================================================

RESULTS_BASE="/home/guilherme/git/pix2pixHD/results"
CENTRAL_ABLATION_DIR="/home/guilherme/git/pix2pixHD/ablation_superposition_studies"
FPS=2
IMG_SIZE=720
MAX_IMAGES=26       # Process up to N frames per experiment (set to 0 for all)
SSIM_THRESH=0.65

# Format: "EXPERIMENT_NAME|PHASE|SAVE_ALL_FRAMES_FLAG"
# Use "--save_all_frames" to save every frame, or leave empty to only save frames where SSIM < 0.75
EXPERIMENTS=(
    "sonar2optical_didson_cleaned|test_latest|--save_all_frames"
    #"sonar2optical_didson_padd|test_latest|--save_all_frames"
    #"sonar2optical_didson_stretch|test_latest|--save_all_frames"
)

mkdir -p "$CENTRAL_ABLATION_DIR"

echo "========================================================="
echo " Starting Red/Green Overlap Ablation Batch "
echo " Saving folder: ${CENTRAL_ABLATION_DIR}"
echo "========================================================="

for EXP_ENTRY in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r EXP_NAME PHASE SAVE_FLAG <<< "$EXP_ENTRY"

    IMG_DIR="${RESULTS_BASE}/${EXP_NAME}/${PHASE}/images"
    OUT_VID="${RESULTS_BASE}/${EXP_NAME}/${EXP_NAME}_overlap_study.mp4"

    echo "---------------------------------------------------------"
    echo "Processing Experiment: ${EXP_NAME}"
    echo "Results Dir:           ${IMG_DIR}"
    echo "---------------------------------------------------------"

    if [ ! -d "$IMG_DIR" ]; then
        echo "Warning: Results directory '${IMG_DIR}' not found. Skipping."
        continue
    fi
    
    CMD=(python3 create_overlap_analysis.py
        --results_dir "$IMG_DIR"
        --output_video "$OUT_VID"
        --output_frames_dir "$CENTRAL_ABLATION_DIR"
        --exp_name "$EXP_NAME"
        --fps "$FPS"
        --img_size "$IMG_SIZE"
        --ssim_threshold "$SSIM_THRESH"
        --use_rembg
    )

    if [ -n "$MAX_IMAGES" ] && [ "$MAX_IMAGES" -gt 0 ]; then
        CMD+=(--max_images "$MAX_IMAGES")
    fi

    if [ -n "$EXTRA_FLAGS" ]; then
        CMD+=($EXTRA_FLAGS)
    fi

    "${CMD[@]}"
done

echo "========================================================="
echo " All ablation studies complete! "
echo " Inspect superposition results in: ${CENTRAL_ABLATION_DIR}"
echo "========================================================="