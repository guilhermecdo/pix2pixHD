#!/bin/bash

# Configuration
RAW_SRC=~/git/pix2pixHD/CroppedImages/CroppedImages
BASE_DATASET_DIR="./datasets/Oculus2Didson"
MODES=("pad" "interp")
SIZES=(128 256)
TRAIN_PERCENTAGES=(0.6 0.7 0.8 0.9)

# Loop through every combination
for MODE in "${MODES[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        for PERC in "${TRAIN_PERCENTAGES[@]}"; do
            
            EXP_NAME="Exp_${MODE}_${SIZE}_p${PERC}"
            CURRENT_DST="${BASE_DATASET_DIR}_${EXP_NAME}"
            
            echo "=========================================================="
            echo "STARTING EXPERIMENT: $EXP_NAME"
            echo "=========================================================="

            # 1. DATA PREPARATION
            # TEST_SPLIT=$(echo "1.0 - $PERC" | bc)
            # python prepare_sonar_data.py --src "$RAW_SRC" --dst "$CURRENT_DST" \
            #                              --mode "$MODE" --size "$SIZE" --split "$TEST_SPLIT"

            # # 2. TRAINING
            # if [ "$SIZE" -eq 128 ]; then BATCH=128; else BATCH=32; fi
            
            # python -u train.py --name "$EXP_NAME" \
            #                    --dataroot "$CURRENT_DST" \
            #                    --no_instance --label_nc 0 --resize_or_crop none \
            #                    --nThreads 0 --gpu_ids 0 --no_vgg_loss \
            #                    --batchSize "$BATCH" --save_epoch_freq 50 --print_freq 50 \
                               
            # # 3. TESTING (INFERENCE)
            # echo "Generating test results for $EXP_NAME..."
            # # Note: test.py will use the 'latest' weights by default
            # python test.py --name "$EXP_NAME" \
            #                --dataroot "$CURRENT_DST" \
            #                --no_instance --label_nc 0 --resize_or_crop none \
            #                --nThreads 0 --gpu_ids 0
            
            # 4. QUANTITATIVE EVALUATION (METRICS)
            echo "Calculating metrics for $EXP_NAME..."
            RESULTS_DIR="./results/${EXP_NAME}/test_latest/images/"
            
            # This calls the metrics script I gave you earlier
            # We use 'python -u' to ensure the metrics are printed to the console/log
            # python -u evaluate_metrics.py --results_dir "$RESULTS_DIR" >> "summary_results.log"

            # Evaluate ONLY the sonar content (original form)
            python evaluate_metrics_original.py --results_dir "$RESULTS_DIR" --mode "$MODE" --orig_w 70 --orig_h 40 >> "summary_original.log"

            echo "FINISHED EXPERIMENT AND EVALUATION: $EXP_NAME"
            echo "----------------------------------------------------------"
        done
    done
done