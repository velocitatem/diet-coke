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

############################
# STEP 1: Train teacher model
############################
echo "Step 1: Training teacher (BERT) model..." | tee -a "$LOG_FILE"
TEACHER_DIR="$BASE_DIR/teacher"
python src/train_teacher.py hydra.run.dir="$TEACHER_DIR" 2>&1 | tee -a "$LOG_FILE"

# Find the actual output directory created by Hydra for the teacher
ACTUAL_TEACHER_DIR=$(find outputs -type d -path "*/teacher*" -newer "$BASE_DIR" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
if [ -z "$ACTUAL_TEACHER_DIR" ]; then
    # Try to find any recent teacher checkpoint
    ACTUAL_TEACHER_DIR=$(dirname $(dirname $(find outputs -name "teacher.ckpt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)))
fi

if [ -z "$ACTUAL_TEACHER_DIR" ]; then
    echo "Could not find teacher output directory" | tee -a "$LOG_FILE"
    exit 1
fi

TEACHER_CHECKPOINT=$(find "$ACTUAL_TEACHER_DIR" -name "teacher.ckpt" -type f)
if [ -z "$TEACHER_CHECKPOINT" ]; then
    echo "Teacher model checkpoint not found in $ACTUAL_TEACHER_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

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

############################
# STEP 2: Distill to Decision Tree
############################
echo "Step 2: Distilling knowledge to Decision Tree..." | tee -a "$LOG_FILE"
STUDENT_DIR="$BASE_DIR/student"
python src/distill_to_tree.py hydra.run.dir="$STUDENT_DIR" paths.teacher_model="$TEACHER_CHECKPOINT" 2>&1 | tee -a "$LOG_FILE"

# Find the actual output directory created by Hydra for the student
ACTUAL_STUDENT_DIR=$(find outputs -type d -path "*/student*" -newer "$TEACHER_DIR" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
if [ -z "$ACTUAL_STUDENT_DIR" ]; then
    # Try to find the student model in a recent directory
    STUDENT_MODEL_PATH=$(find outputs -path "*/artifacts/student.pkl" -type f -newer "$TEACHER_DIR" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
    if [ -n "$STUDENT_MODEL_PATH" ]; then
        ACTUAL_STUDENT_DIR=$(dirname $(dirname "$STUDENT_MODEL_PATH"))
    fi
fi

if [ -z "$ACTUAL_STUDENT_DIR" ]; then
    echo "Could not find student output directory" | tee -a "$LOG_FILE"
    exit 1
fi

STUDENT_MODEL=$(find "$ACTUAL_STUDENT_DIR" -name "student.pkl" -type f)
if [ -z "$STUDENT_MODEL" ]; then
    echo "Student model not found in $ACTUAL_STUDENT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# Check if distillation succeeded
if [ ! -f "$STUDENT_MODEL" ]; then
    echo "Distillation failed. Student model not found at $STUDENT_MODEL" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Student model successfully saved at $STUDENT_MODEL" | tee -a "$LOG_FILE"

############################
# STEP 3: Evaluate both models
############################
echo "Step 3: Evaluating models..." | tee -a "$LOG_FILE"
EVAL_DIR="$BASE_DIR/eval"
python src/evaluate.py hydra.run.dir="$EVAL_DIR" paths.teacher_model="$TEACHER_CHECKPOINT" paths.student_model="$STUDENT_MODEL" 2>&1 | tee -a "$LOG_FILE"

# Find the actual evaluation directory
ACTUAL_EVAL_DIR=$(find outputs -type d -path "*/eval*" -newer "$STUDENT_DIR" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
if [ -z "$ACTUAL_EVAL_DIR" ]; then
    # Try to find the evaluation report in a recent directory
    EVAL_REPORT_PATH=$(find outputs -name "evaluation_report.json" -type f -newer "$STUDENT_DIR" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
    if [ -n "$EVAL_REPORT_PATH" ]; then
        ACTUAL_EVAL_DIR=$(dirname "$EVAL_REPORT_PATH")
    else
        ACTUAL_EVAL_DIR="$EVAL_DIR"
    fi
fi

EVAL_REPORT="$ACTUAL_EVAL_DIR/evaluation_report.json"

############################
# Final summary
############################
echo "Pipeline completed successfully!" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "- Teacher model: $TEACHER_CHECKPOINT" | tee -a "$LOG_FILE"
echo "- Student model: $STUDENT_MODEL" | tee -a "$LOG_FILE"
echo "- Evaluation: $EVAL_REPORT" | tee -a "$LOG_FILE"
echo "To view TensorBoard logs: tensorboard --logdir $ACTUAL_TEACHER_DIR/tensorboard" | tee -a "$LOG_FILE"

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