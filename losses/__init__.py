"""Loss function factory."""

import torch.nn as nn
from losses.rate_distortion import RateDistortionLoss
from losses.perceptual import PerceptualLoss
from losses.semantic import SemanticCrossEntropyLoss


def get_loss(name: str, **kwargs):
    """Create a loss function by name."""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "rate_distortion":
        return RateDistortionLoss(**kwargs)
    elif name == "perceptual":
        return PerceptualLoss(**kwargs)
    elif name == "cross_entropy":
        return SemanticCrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}. "
                         "Available: mse, l1, rate_distortion, perceptual, cross_entropy")
