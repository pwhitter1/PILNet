#!/bin/bash

# Exit script on any error
set -e

# Check whether conda is available
if ! command -v conda &> /dev/null; then
  echo "Error: conda not found. Please install Anaconda."
  exit 1
fi

# Set environment name
ENV_NAME=env_pilnet_pkg

# Create new conda environment
echo "Creating conda environment '$ENV_NAME' with Python 3.9..."
conda create -n "$ENV_NAME" -y -c conda-forge python=3.9 numpy=1.21.6 h5py=3.7.0 rdkit==2022.9.4

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install package dependencies
echo "Installing package dependencies..."
echo "[Assumes CUDA Version 12.1]"

conda install -y -c conda-forge psi4

pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 \
  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html \

echo ""
echo "All packages installed successfully in environment '$ENV_NAME'."
