"""
High-Performance Hardware and Distributed Scaling for GPUs & Google TPUs in JAX.

Provides:
  1. Hardware detection & environment setup (TPU, NVIDIA/AMD GPU, CPU)
  2. Mesh construction and NamedSharding with PartitionSpec
  3. Seamless single-device and multi-device distributed execution
  4. Automatic XLA performance flags
"""

import os
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def setup_xla_environment(
    enable_gpu_memory_fraction: Optional[float] = 0.90,
    platform: Optional[str] = None,
):
    """
    Configure high-performance environment flags for XLA, GPU, and TPU.
    """
    # Prevent JAX from pre-allocating 100% of GPU memory if specified
    if enable_gpu_memory_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(enable_gpu_memory_fraction)

    # Enable XLA flags for aggressive kernel fusion
    current_xla_flags = os.environ.get("XLA_FLAGS", "")
    flags = [
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_triton_gemm=true",
    ]
    for flag in flags:
        if flag not in current_xla_flags:
            current_xla_flags = f"{current_xla_flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = current_xla_flags

    if platform:
        jax.config.update("jax_platform_name", platform)


def detect_hardware() -> Dict[str, Any]:
    """
    Introspect active JAX hardware platform, device count, and capabilities.

    Returns:
        Dictionary with backend ('tpu', 'gpu', or 'cpu'), device count, and device objects.
    """
    backend = jax.default_backend()
    devices = jax.devices()
    num_devices = len(devices)
    device_kinds = [d.device_kind for d in devices]

    is_tpu = backend == "tpu" or any("tpu" in k.lower() for k in device_kinds)
    is_gpu = backend == "gpu" or any("cuda" in k.lower() or "gpu" in k.lower() for k in device_kinds)
    is_cpu = backend == "cpu" and not (is_tpu or is_gpu)

    target_name = "Cloud TPU" if is_tpu else ("NVIDIA/AMD GPU" if is_gpu else "Host CPU")

    return {
        "backend": backend,
        "target_name": target_name,
        "is_tpu": is_tpu,
        "is_gpu": is_gpu,
        "is_cpu": is_cpu,
        "num_devices": num_devices,
        "devices": devices,
        "device_kinds": device_kinds,
    }


def create_device_mesh(
    data_axis: int = -1,
    model_axis: int = 1,
) -> Tuple[Optional[Mesh], Dict[str, NamedSharding]]:
    """
    Construct a modern JAX Mesh for distributed scaling across GPUs or TPUs.

    Args:
        data_axis: Number of devices for Data Parallelism (DP) (-1 = auto-calculate)
        model_axis: Number of devices for Model Parallelism (MP) (1 = standard data parallel)

    Returns:
        mesh: jax.sharding.Mesh
        shardings: Dict of pre-configured NamedSharding instances
    """
    devices = jax.devices()
    num_devices = len(devices)

    if model_axis == 1:
        data_axis = num_devices
    elif data_axis == -1:
        data_axis = num_devices // model_axis

    if data_axis * model_axis != num_devices:
        # Fallback to 1D data parallel mesh
        device_grid = np.array(devices).reshape((num_devices, 1))
        mesh = Mesh(device_grid, axis_names=("data", "model"))
    else:
        device_grid = np.array(devices).reshape((data_axis, model_axis))
        mesh = Mesh(device_grid, axis_names=("data", "model"))

    # Sharding definitions:
    # 1. Replicated (all devices hold a complete copy, e.g. weights in standard DP)
    replicated = NamedSharding(mesh, P())

    # 2. Data-parallel batch sharding (inputs split along batch dimension across 'data' axis)
    batch_sharding = NamedSharding(mesh, P("data", None))

    # 3. Model-parallel sharding (e.g. for FSDP / tensor parallel weight splitting)
    model_sharding = NamedSharding(mesh, P("model", None))

    shardings = {
        "replicated": replicated,
        "batch": batch_sharding,
        "model": model_sharding,
    }

    return mesh, shardings
