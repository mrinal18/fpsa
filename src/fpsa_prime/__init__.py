"""FPSA-Prime: pure attention-equilibrium reasoning."""

from .attention import DualBankFixedPointAttention, FPSAContext
from .config import ARCH_PRESETS, FPSAPrimeConfig, build_config
from .model import FPSAPrimeReasoner, build_model
from .stability import local_jacobian_spectral_penalty

__all__ = [
    "ARCH_PRESETS",
    "DualBankFixedPointAttention",
    "FPSAContext",
    "FPSAPrimeConfig",
    "FPSAPrimeReasoner",
    "build_config",
    "build_model",
    "local_jacobian_spectral_penalty",
]
