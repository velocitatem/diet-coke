#!/bin/bash
set -e

# Script to set up the environment for BERT to Decision Tree distillation

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies if specified
if [ "$1" == "--dev" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install the package in development mode
echo "Installing the package in development mode..."
pip install -e .

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration if needed."
fi

# Create directories for outputs
mkdir -p outputs

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the full pipeline, use: ./scripts/run_full_pipeline.sh" 