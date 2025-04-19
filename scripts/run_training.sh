#!/bin/bash
set -e

# Script to run just the teacher model training

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

echo "Starting teacher model training..."
echo "Output will be saved to $OUTPUT_DIR"

# Run teacher model training
python src/train_teacher.py hydra.run.dir="$OUTPUT_DIR"

# Verify the checkpoint was created
if [ -f "$OUTPUT_DIR/checkpoints/teacher.ckpt" ]; then
    FILE_SIZE=$(stat -c%s "$OUTPUT_DIR/checkpoints/teacher.ckpt")
    echo "Training completed successfully!"
    echo "Teacher model checkpoint saved to $OUTPUT_DIR/checkpoints/teacher.ckpt (${FILE_SIZE} bytes)"
else
    echo "Training failed. Checkpoint not found at $OUTPUT_DIR/checkpoints/teacher.ckpt"
    exit 1
fi 