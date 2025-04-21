#!/bin/bash
set -e

# Script to run just the teacher model training

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create output directories
mkdir -p outputs/teacher/checkpoints
mkdir -p outputs/student/artifacts

echo "Starting teacher model training..."
python src/train_teacher.py

echo "Starting distillation process..."
python src/distill_to_tree.py

echo "Starting evaluation..."
python src/evaluate.py

# Verify the checkpoint was created
if [ -f "outputs/teacher/checkpoints/teacher.ckpt" ]; then
    FILE_SIZE=$(stat -c%s "outputs/teacher/checkpoints/teacher.ckpt")
    echo "Training completed successfully!"
    echo "Teacher model checkpoint saved to outputs/teacher/checkpoints/teacher.ckpt (${FILE_SIZE} bytes)"
else
    echo "Training failed. Checkpoint not found at outputs/teacher/checkpoints/teacher.ckpt"
    exit 1
fi 