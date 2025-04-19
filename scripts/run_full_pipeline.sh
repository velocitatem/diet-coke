#!/bin/bash
set -e

# Script to run the full BERT to Decision Tree distillation pipeline

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

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
BASE_DIR="outputs/$TIMESTAMP"
mkdir -p "$BASE_DIR"

# Log setup
LOG_FILE="$BASE_DIR/pipeline.log"
echo "Starting pipeline run at $(date)" | tee -a "$LOG_FILE"
echo "Base directory: $BASE_DIR" | tee -a "$LOG_FILE"

# Step 1: Train teacher model
echo "Step 1: Training teacher (BERT) model..." | tee -a "$LOG_FILE"
python src/train_teacher.py hydra.run.dir="$BASE_DIR/teacher" 2>&1 | tee -a "$LOG_FILE"

# Get the actual output directory created by train_teacher.py (it might include the real timestamp)
if [ -d "$BASE_DIR/teacher" ]; then
    TEACHER_OUTPUT_DIR="$BASE_DIR/teacher"
else
    # Try to find the most recent teacher directory using the pattern
    TEACHER_OUTPUT_DIR=$(find outputs -type d -name "*" -printf "%T@ %p\n" | sort -nr | grep -m 1 -v "$BASE_DIR" | cut -d' ' -f2-)
    if [ -z "$TEACHER_OUTPUT_DIR" ]; then
        echo "Could not find teacher output directory" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

TEACHER_CHECKPOINT="$TEACHER_OUTPUT_DIR/checkpoints/teacher.ckpt"

# Check if training succeeded
if [ ! -f "$TEACHER_CHECKPOINT" ]; then
    echo "Teacher model training failed. Checkpoint not found at $TEACHER_CHECKPOINT" | tee -a "$LOG_FILE"
    exit 1
fi

# Verify checkpoint file size to ensure it's not empty
FILE_SIZE=$(stat -c%s "$TEACHER_CHECKPOINT" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 1000000 ]; then  # Minimum expected size (1MB)
    echo "Teacher model checkpoint seems too small or corrupted (${FILE_SIZE} bytes). Training may have failed." | tee -a "$LOG_FILE"
    exit 1
fi

echo "Teacher model checkpoint successfully saved at $TEACHER_CHECKPOINT (${FILE_SIZE} bytes)" | tee -a "$LOG_FILE"

# Step 2: Distill to Decision Tree
echo "Step 2: Distilling knowledge to Decision Tree..." | tee -a "$LOG_FILE"
python src/distill_to_tree.py hydra.run.dir="$BASE_DIR/student" paths.teacher_model="$TEACHER_CHECKPOINT" 2>&1 | tee -a "$LOG_FILE"

# Get the actual student output directory
STUDENT_OUTPUT_DIR="$BASE_DIR/student"
STUDENT_MODEL="$STUDENT_OUTPUT_DIR/artifacts/student.pkl"

# Check if distillation succeeded
if [ ! -f "$STUDENT_MODEL" ]; then
    echo "Distillation failed. Student model not found at $STUDENT_MODEL" | tee -a "$LOG_FILE"
    exit 1
fi

# Step 3: Evaluate both models
echo "Step 3: Evaluating models..." | tee -a "$LOG_FILE"
python src/evaluate.py hydra.run.dir="$BASE_DIR/eval" paths.teacher_model="$TEACHER_CHECKPOINT" paths.student_model="$STUDENT_MODEL" 2>&1 | tee -a "$LOG_FILE"

EVAL_REPORT="$BASE_DIR/eval/evaluation_report.json"

# Final summary
echo "Pipeline completed successfully!" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "- Teacher model: $TEACHER_CHECKPOINT" | tee -a "$LOG_FILE"
echo "- Student model: $STUDENT_MODEL" | tee -a "$LOG_FILE"
echo "- Evaluation: $EVAL_REPORT" | tee -a "$LOG_FILE"
echo "To view TensorBoard logs: tensorboard --logdir $TEACHER_OUTPUT_DIR/tensorboard" | tee -a "$LOG_FILE"

# Print final metrics summary
if [ -f "$EVAL_REPORT" ]; then
    echo "Final metrics:" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    jq '.teacher.metrics.accuracy, .student.metrics.accuracy, .fidelity.test_agreement' "$EVAL_REPORT" | \
    paste <(echo -e "Teacher accuracy:\nStudent accuracy:\nTeacher-Student agreement:") - | \
    column -t | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
else
    echo "Evaluation report not found at $EVAL_REPORT. Check logs for errors." | tee -a "$LOG_FILE"
fi 