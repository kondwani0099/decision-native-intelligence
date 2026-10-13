# High-Performance JAX / UDM-200M Container
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Environment flags
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
ENV XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=true"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for rapid dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN uv pip install --system --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    uv pip install --system -r requirements.txt

# Copy codebase
COPY . /app

EXPOSE 8000

# Default command: run inference verification
CMD ["python3", "infer_jax.py"]
