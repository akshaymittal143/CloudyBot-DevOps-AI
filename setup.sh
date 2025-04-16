#!/bin/bash

# Deactivate existing virtual environment if any
deactivate 2>/dev/null

# Remove existing virtual environment
rm -rf venv

# Create new virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install numpy first to ensure correct version
pip install numpy==1.24.3

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install remaining requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p .streamlit

echo "Setup completed successfully!"