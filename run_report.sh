#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# CONFIGURATION
# ==========================================
EXP_NAME="holoocean-didsion"
BASE_DATASET_DIR="./datasets/holoocean-didson-128"

# Resolve absolute locations for safety
RESULTS_DIR="./results/${EXP_NAME}/test_latest/images"

echo "========================================================="
echo "   Starting Sonar Post-Processing & Reporting Pipeline   "
echo "========================================================="
echo "Experiment Target: $EXP_NAME"
echo "Dataset Directory: $BASE_DATASET_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "---------------------------------------------------------"

# Ensure results exist before running reports
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory does not exist!"
    echo "Did you finish running 'python3 test.py' for this model?"
    exit 1
fi

# ==========================================
# STEP 1: Build Three-Column Final Report
# ==========================================
if [ -f "build_final_report.py" ]; then
    echo "[STEP 1/3] Building primary HTML Visual Report..."
    python3 build_final_report.py \
        --results_dir "$RESULTS_DIR" \
        --dataset_dir "$BASE_DATASET_DIR"
else
    echo "[SKIP] build_final_report.py not found in current directory."
fi

echo "---------------------------------------------------------"

# ==========================================
# STEP 2: Trigger Interactive Curation (If needed)
# ==========================================
# Since you have the Tkinter scripts to select good vs bad results
if [ -f "curate_results_jet.py" ]; then
    echo "[STEP 2/3] Launching Interactive JET Curation Interface..."
    echo "-> Select this experiment inside the window menu to begin filtering."
    python3 curate_results_jet.py
elif [ -f "curate_results.py" ]; then
    echo "[STEP 2/3] Launching Interactive Curation Interface..."
    python3 curate_results.py
else
    echo "[SKIP] Curation GUI scripts not found."
fi

echo "---------------------------------------------------------"
echo "Pipeline completed successfully!"
echo "Check your results under: $RESULTS_DIR/index.html"
echo "========================================================="