# #!/bin/bash
# set -e

# ==========================================
# 1. CONFIGURATION
# ==========================================
RAW_DATA_ROOT="./datasets/Sonar2Optical"
STAGE_DATA_ROOT="./datasets/sonar2optical_matched"
EXP_NAME="sonar2optical_exp"

# echo "========================================================="
# echo "   Starting Sonar to Optical Master Training Pipeline   "
# echo "========================================================="

# # ==========================================
# # 2. PREPARE DATASTAGE (RESOLVE NAMING MISMATCH)
# # ==========================================
# echo "[STEP 1/4] Creating matched symlinks for pix2pixHD..."

# # Clean old staging directory if it exists to avoid mixing experiments
# rm -rf "$STAGE_DATA_ROOT"

# # Loop through both training and testing phases
# for PHASE in "train" "test"; do
#     src_A="${RAW_DATA_ROOT}/${PHASE}_A"
#     src_B="${RAW_DATA_ROOT}/${PHASE}_B"
    
#     dst_A="${STAGE_DATA_ROOT}/${PHASE}_A"
#     dst_B="${STAGE_DATA_ROOT}/${PHASE}_B"
    
#     # Create output directories
#     mkdir -p "$dst_A"
#     mkdir -p "$dst_B"
    
#     # Verify raw directories exist
#     if [ ! -d "$src_A" ] || [ ! -d "$src_B" ]; then
#         echo "Skipping phase ${PHASE}: Raw folders not found."
#         continue
#     fi

#     echo "Processing ${PHASE} split..."
    
#     # Read files into arrays, sorting them to ensure alignments match up
#     # Using find to handle both .jpg and .jpeg seamlessly
#     files_A=($(find "$src_A" -maxdepth 1 -type f \( -name "Sonar_*" \) | sort))
#     files_B=($(find "$src_B" -maxdepth 1 -type f \( -name "RGB_*" \) | sort))

#     count_A=${#files_A[@]}
#     count_B=${#files_B[@]}

#     # Ensure we have matching counts
#     if [ "$count_A" -ne "$count_B" ]; then
#         echo "Warning: Mismatched counts in ${PHASE}! A: $count_A files, B: $count_B files. Using minimum."
#     fi

#     # Determine maximum valid index loop limit
#     limit=$(( count_A < count_B ? count_A : count_B ))

#     for ((i=0; i<limit; i++)); do
#         file_A="${files_A[$i]}"
#         file_B="${files_B[$i]}"
        
#         # Extract file extension dynamically (.jpg, .png, etc.)
#         ext="${file_A##*.}"
        
#         # Enforce identical uniform names using the loop index (e.g., frame_00001.jpg)
#         padded_idx=$(printf "%05d" $i)
#         new_filename="frame_${padded_idx}.${ext}"
        
#         # Create absolute paths for symlinks to prevent breaking path links
#         ln -s "$(canvas_p=$(cd "$(dirname "$file_A")" && pwd) && echo "$canvas_p/$(basename "$file_A")")" "${dst_A}/${new_filename}"
#         ln -s "$(canvas_p=$(cd "$(dirname "$file_B")" && pwd) && echo "$canvas_p/$(basename "$file_B")")" "${dst_B}/${new_filename}"
#     done
#     echo "Successfully linked $limit matched pairs for ${PHASE} split."
# done

# echo "---------------------------------------------------------"

# ==========================================
# 3. RUN TRAINING
# ==========================================
# echo "[STEP 2/4] Initializing Network Training..."

# python3 train.py --name "$EXP_NAME" \
#   --dataroot "$STAGE_DATA_ROOT" \
#   --no_instance --label_nc 0 \
#   --resize_or_crop resize_and_crop --loadSize 512 --fineSize 512 \
#   --nThreads 0 --gpu_ids 0 --no_vgg_loss \
#   --batchSize 4 --save_epoch_freq 5 --print_freq 5 \
#   --niter 100 --niter_decay 100
# echo "---------------------------------------------------------"

# ==========================================
# 4. RUN TESTING / INFERENCE
# ==========================================
echo "[STEP 3/4] Running Inference over Test Dataset Split..."

# Ensure --resize_or_crop, --loadSize, and --fineSize match training exactly!
python3 test2.py --name "$EXP_NAME" \
  --dataroot "$STAGE_DATA_ROOT" \
  --no_instance --label_nc 0 \
  --resize_or_crop resize_and_crop --loadSize 512 --fineSize 512 \
  --nThreads 0 --gpu_ids 0 \
  --how_many 2000

echo "---------------------------------------------------------"

# ==========================================
# 5. RUN EVALUATION REPORTS
# ==========================================
echo "[STEP 4/4] Generating Visual Evaluation Reports..."
RESULTS_DIR="./results/${EXP_NAME}/test_latest/images"

if [ -f "build_final_report.py" ]; then
    python3 build_final_report.py \
        --results_dir "$RESULTS_DIR" \
        --dataset_dir "$STAGE_DATA_ROOT"
    echo "Pipeline complete! Visual report generated at: ${RESULTS_DIR}/index.html"
else
    echo "Warning: build_final_report.py not found in root. Reports generation skipped."
fi
echo "========================================================="