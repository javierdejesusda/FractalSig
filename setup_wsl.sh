#!/bin/bash

# FractalSig - WSL2/Linux Environment Setup Script
# This script sets up the hybrid JAX + PyTorch environment with CUDA acceleration.

set -e

ENV_NAME="fractalsig"

echo "   FractalSig Environment Setup (WSL2/Linux)        "

# Create Conda Environment
echo "[1/5] Creating Conda environment: $ENV_NAME..."
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Updating..."
    conda env update -n $ENV_NAME -f environment.yaml
else
    conda env create -f environment.yaml
fi

# Activate Environment
# Note: Source conda.sh to ensure 'conda activate' works in the script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Handle iisignature Compilation
echo "[2/5] Installing iisignature with --no-build-isolation..."
# We use --no-build-isolation because iisignature needs to see the 
# host environment's build tools (numpy, setuptools) to compile its C extensions correctly on WSL2.
pip install iisignature==0.24 --no-build-isolation

# Install JAX with CUDA support
echo "[3/5] Installing JAX with CUDA 12 support..."
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Finalize Dependencies
echo "[4/5] Syncing remaining dependencies from requirements.txt..."
pip install -r requirements.txt

# Verification
echo "[5/5] Running verification..."
python << END
import torch
import jax
import iisignature
import numpy as np

print(f"Torch Version: {torch.__version__}")
print(f"Torch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"JAX Version: {jax.__version__}")
print(f"JAX Devices: {jax.devices()}")
print(f"iisignature: Installed and ready")

if torch.cuda.is_available() and str(jax.devices()[0]).startswith('gpu'):
    print("SUCCESS: Unified CUDA Acceleration Verified for both PyTorch and JAX.")
else:
    print("WARNING: GPU not detected by one or more frameworks. Check drivers/CUDA versions.")
END

echo "Setup Complete! Active environment with: conda activate $ENV_NAME"
