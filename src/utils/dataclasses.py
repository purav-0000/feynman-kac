from dataclasses import dataclass
from typing import Optional, Any
import torch

@dataclass
class CollocationData:
    """Standard PINN training data."""
    X_u: torch.Tensor
    X_f: torch.Tensor
    u: torch.Tensor


@dataclass
class PathData:
    """Spatiotemporal paths for SINDy and fine-tuning."""
    s: Any
    t: Any
    u: Any


@dataclass
class DerivativeData:
    """Predicted derivatives from the neural network."""
    u_t: Any
    u_s: Any
    u_ss: Any
    u: Optional[Any] = None


@dataclass
class SINDyContext:
    """Stochastic parameters and configuration for SINDy."""
    assumed_R: float
    uniform_t: Optional[bool] = None
    dt: Optional[float] = None
    sigma_est: Optional[Any] = None
    recovered_dB: Optional[Any] = None
    mask_indices: Optional[list] = None