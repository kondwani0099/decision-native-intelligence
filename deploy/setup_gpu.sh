#!/bin/bash
# ==============================================================================
# Setup High-Performance JAX with CUDA 12 for NVIDIA GPUs (RTX / A100 / H100)
# ==============================================================================

set -e

echo "=== Setting up JAX for NVIDIA GPU (CUDA 12) ==="

# 1. Update pip and uv
python3 -m pip install --upgrade pip uv

# 2. Install JAX with CUDA 12 support
uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install Flax, Optax, and Decision Framework dependencies
uv pip install --upgrade flax optax numpy scipy scikit-learn matplotlib pyyaml

# 4. Set High-Performance XLA Environment Variables
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true

echo "=== Verifying JAX GPU Backend ==="
python3 -c "import jax; print('JAX version:', jax.__version__); print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"

echo "=== JAX GPU Setup Complete! ==="
