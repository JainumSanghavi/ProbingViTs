"""Auto-detect best available device: MPS > CUDA > CPU."""

import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Return device information for logging."""
    device = get_device()
    info = {"device": str(device)}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
    return info
