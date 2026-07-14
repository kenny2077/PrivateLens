#!/bin/bash
# Setup script for PrivateLens

set -e

echo "Setting up PrivateLens..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e "."

# Create data directory
mkdir -p ~/.privatelens/thumbnails
mkdir -p ~/.privatelens/models

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Scan a folder:  privatelens scan ~/Pictures"
echo "  2. Index photos:   privatelens index"
echo "  3. Search:         privatelens search 'driver license'"
echo "  4. Web UI:         privatelens serve"
