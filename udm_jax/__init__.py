"""
UDM / JEV Model in JAX.

High-performance decision-native intelligence framework targeting GPUs and Google TPUs.
Includes UDM-200M (201.4M parameter) model architecture.
"""

from .config import UDMJaxConfig, UDM_200M, UDM_125M, UDM_350M, UDM_1B

try:
    from .model.udm import UDMJaxModel
    from .model.decision_heads import JaxDecisionOutput
    from .distributed import detect_hardware, create_device_mesh, setup_xla_environment
    from .losses import compute_combined_loss
except ImportError:
    pass

__all__ = [
    "UDMJaxConfig",
    "UDM_200M",
    "UDM_125M",
    "UDM_350M",
    "UDM_1B",
    "UDMJaxModel",
    "JaxDecisionOutput",
    "detect_hardware",
    "create_device_mesh",
    "setup_xla_environment",
    "compute_combined_loss",
]
