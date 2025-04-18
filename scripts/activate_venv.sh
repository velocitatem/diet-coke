#!/bin/bash
# Script to activate the virtual environment

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Virtual environment not found. Please run ./scripts/setup_env.sh first."
    exit 1
fi

# Print activation command (for use with source)
echo "# Run this command to activate the virtual environment:"
echo "source $PROJECT_ROOT/venv/bin/activate"

# If this script is sourced (not executed), activate the environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Virtual environment activated."
fi 