"""Reproducibility helpers shared by formal experiment runners."""
from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> int:
    """Seed every RNG used by the formal stack and return the normalized seed."""
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def reproducibility_metadata(seed: int, device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_name = None
    if cuda_available:
        try:
            cuda_name = torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            cuda_name = None
    return {
        "seed": int(seed),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_name,
        "device": str(device),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
