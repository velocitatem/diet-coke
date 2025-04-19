#!/bin/bash
set -e

# Script to run the distillation part of the pipeline with the latest teacher checkpoint

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create timestamp for this run
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="outputs/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

# Find the latest teacher checkpoint
LATEST_TEACHER=$(find outputs -name "teacher.ckpt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)

if [ -z "$LATEST_TEACHER" ]; then
    echo "No teacher checkpoint found. Please run train_teacher.py first."
    exit 1
fi

# Check if the file exists and has a reasonable size
if [ ! -f "$LATEST_TEACHER" ]; then
    echo "Teacher checkpoint file not found at $LATEST_TEACHER"
    exit 1
fi

FILE_SIZE=$(stat -c%s "$LATEST_TEACHER" 2>/dev/null || echo "0")
if [ "$FILE_SIZE" -lt 1000000 ]; then  # Minimum expected size (1MB)
    echo "Teacher model checkpoint seems too small or corrupted (${FILE_SIZE} bytes)."
    exit 1
fi

echo "Found latest teacher checkpoint: $LATEST_TEACHER (${FILE_SIZE} bytes)"
echo "Output will be saved to $OUTPUT_DIR"

# Run distillation
echo "Starting distillation process..."
python src/distill_to_tree.py hydra.run.dir="$OUTPUT_DIR" paths.teacher_model="$LATEST_TEACHER"

if [ -f "$OUTPUT_DIR/artifacts/student.pkl" ]; then
    echo "Distillation completed successfully!"
    echo "Student model saved to $OUTPUT_DIR/artifacts/student.pkl"
else
    echo "Distillation failed. Student model not found at $OUTPUT_DIR/artifacts/student.pkl"
    exit 1
fi 