#!/bin/bash
set -e

# ==========================================================
# Batch Metric Evaluation Runner (SSIM, PSNR, LPIPS)
# ==========================================================

CHECKPOINTS_DIR="/home/guilherme/git/pix2pixHD/checkpoints"
DATA_ROOT_BASE="/home/guilherme/git/pix2pixHD/datasets"

# Define your target runs in format: "EXPERIMENT_NAME|DATASET_SUBDIR|HOW_MANY_SAMPLES"
EXPERIMENTS=(
    #"sonar2optical_didson_cleaned|DidsonFullSet|200"
    "didson_cleaned_G_4|DidsonFullSet_Cleaned_2|26"
    # "6%_Data|Loops/Loop1+2|200"
    # "25%_Data|Loops/Loop1+2|200"
    # "50%_Data|Loops/Loop1+2|200"
    #"DidsonFull|DidsonFullSet|200"
    #"sonar2optical_loop_3|Loops/Loop3|200"
    #"sonar2optical_loop_3_no_transfer|Loops/Loop3|200"
    #"sonar2optical_loop_1+2|Loops/Loop1+2|200"
)

for EXP_ENTRY in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r EXP_NAME DATA_SUBDIR HOW_MANY <<< "$EXP_ENTRY"
    DATA_PATH="${DATA_ROOT_BASE}/${DATA_SUBDIR}"
    
    echo "========================================================="
    echo "Starting evaluation for: ${EXP_NAME}"
    echo "Dataset: ${DATA_PATH}"
    echo "Samples: ${HOW_MANY}"
    echo "========================================================="

    if [ ! -d "${CHECKPOINTS_DIR}/${EXP_NAME}" ]; then
        echo "Warning: Checkpoint directory ${CHECKPOINTS_DIR}/${EXP_NAME} not found. Skipping."
        continue
    fi

    python3 metrics2.py \
        --name "$EXP_NAME" \
        --checkpoints_dir "$CHECKPOINTS_DIR" \
        --dataroot "$DATA_PATH" \
        --no_instance \
        --label_nc 0 \
        --resize_or_crop resize_and_crop \
        --loadSize 720 \
        --fineSize 720 \
        --how_many "$HOW_MANY" \
        --no_rembg

    echo "Completed evaluation for ${EXP_NAME}."
    echo ""
done

echo "All batch metric runs completed successfully."