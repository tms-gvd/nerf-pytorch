"""Utilitary functions for NeRF."""
import torch

def get_device():
    """Get the device on which to run.

    Returns:
        torch.device: The device on which to run.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device: ", device)
    return device