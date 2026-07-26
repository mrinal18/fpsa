"""FPSA-R: Fixed-Point Self-Attention Reasoners.

In-layer attention fixed points (FPSA) inside a looped reasoning recursion
(FPRM), lifted to a single joint equilibrium and trained with O(1)-memory
implicit differentiation instead of backpropagation through time.
"""

from .config import ARCH_PRESETS, FPSARConfig, build_config
from .model import FPSAReasoner, build_model

__all__ = ["FPSARConfig", "build_config", "ARCH_PRESETS",
           "FPSAReasoner", "build_model"]
