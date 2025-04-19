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
LATEST_TEACHER=$(find outputs -name "teacher.ckpt" -type f -printf "%T@ %p\n" | sort -nr | head -n1 | cut -d' ' -f2-)

if [ -z "$LATEST_TEACHER" ]; then
    echo "No teacher checkpoint found. Please run train_teacher.py first."
    exit 1
fi

echo "Found latest teacher checkpoint: $LATEST_TEACHER"
echo "Output will be saved to $OUTPUT_DIR"

# Run distillation
python src/distill_to_tree.py hydra.run.dir="$OUTPUT_DIR" paths.teacher_model="$LATEST_TEACHER"

echo "Distillation completed! Results saved to $OUTPUT_DIR" 