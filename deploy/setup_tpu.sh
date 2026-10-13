#!/bin/bash
# ==============================================================================
# Setup High-Performance JAX for Google Cloud TPUs (v2, v3, v4, v5e, v5p, v6e)
# ==============================================================================

set -e

echo "=== Setting up JAX on Google Cloud TPU VM ==="

# 1. Update pip and uv
python3 -m pip install --upgrade pip uv

# 2. Install JAX with official Google libtpu wheel
uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. Install Flax, Optax, and Decision Framework dependencies
uv pip install --upgrade flax optax numpy scipy scikit-learn matplotlib pyyaml

# 4. Optional TPU Pod multi-host coordinator initialization
# export TPU_NAME="<your-tpu-name>"
# export ZONE="<your-gcp-zone>"

echo "=== Verifying JAX TPU Backend ==="
python3 -c "import jax; print('JAX version:', jax.__version__); print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"

echo "=== JAX Cloud TPU Setup Complete! ==="
