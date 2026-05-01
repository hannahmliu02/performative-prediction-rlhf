"""Device selection utilities.

Preference order: CUDA → MPS (Apple Silicon) → CPU.
Used for inference-time model placement; TRL/Accelerate handles training devices
automatically when device_map=None is passed to from_pretrained.
"""

from __future__ import annotations

import torch


def get_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def move_model_to_device(model: torch.nn.Module, device_map: str | None) -> torch.nn.Module:
    """Move *model* to the best device when device_map is None (inference use).

    When device_map is set (e.g. 'auto'), HuggingFace handles placement and
    this function is a no-op.
    """
    if device_map is not None:
        return model
    device = get_device()
    return model.to(device)
