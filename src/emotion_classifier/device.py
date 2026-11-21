import torch

def pick_device() -> str:
    """cuda if available, else mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
