#!/bin/bash
set -e

# Script to run the full BERT to Decision Tree distillation pipeline

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate the virtual environment
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup script first..."
    ./scripts/setup_env.sh
fi
source venv/bin/activate

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Make scripts executable
chmod +x scripts/*.sh

# Create timestamp for this run
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="outputs/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Log setup
LOG_FILE="$OUTPUT_DIR/pipeline.log"
echo "Starting pipeline run at $(date)" | tee -a "$LOG_FILE"
echo "Output will be saved to $OUTPUT_DIR" | tee -a "$LOG_FILE"

# Step 1: Train teacher model
echo "Step 1: Training teacher (BERT) model..." | tee -a "$LOG_FILE"
python src/train_teacher.py hydra.run.dir="$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# Check if training succeeded
if [ ! -f "$OUTPUT_DIR/checkpoints/teacher.ckpt" ]; then
    echo "Teacher model training failed. Check logs for errors." | tee -a "$LOG_FILE"
    exit 1
fi

# Step 2: Distill to Decision Tree
echo "Step 2: Distilling knowledge to Decision Tree..." | tee -a "$LOG_FILE"
python src/distill_to_tree.py hydra.run.dir="$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# Check if distillation succeeded
if [ ! -f "$OUTPUT_DIR/artifacts/student.pkl" ]; then
    echo "Distillation failed. Check logs for errors." | tee -a "$LOG_FILE"
    exit 1
fi

# Step 3: Evaluate both models
echo "Step 3: Evaluating models..." | tee -a "$LOG_FILE"
python src/evaluate.py hydra.run.dir="$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# Final summary
echo "Pipeline completed successfully!" | tee -a "$LOG_FILE"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "To view TensorBoard logs: tensorboard --logdir $OUTPUT_DIR/tensorboard" | tee -a "$LOG_FILE"

# Print final metrics summary
if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
    echo "Final metrics:" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    jq '.teacher.metrics.accuracy, .student.metrics.accuracy, .fidelity.test_agreement' "$OUTPUT_DIR/evaluation_report.json" | \
    paste <(echo -e "Teacher accuracy:\nStudent accuracy:\nTeacher-Student agreement:") - | \
    column -t | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
else
    echo "Evaluation report not found. Check logs for errors." | tee -a "$LOG_FILE"
fi 