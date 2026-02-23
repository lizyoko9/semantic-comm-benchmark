"""Abstract base model for semantic communication."""

import torch
import torch.nn as nn
from abc import abstractmethod


class BaseSemanticModel(nn.Module):
    """Abstract base for all semantic communication models.

    Interface:
        encode(x, **kwargs) → z            (source → channel symbols)
        decode(z, **kwargs) → x_hat        (received → reconstruction)
        power_normalize(z) → z             (avg power = 1)
        forward(x, snr_db) → x_hat         (full pipeline)
    """

    def __init__(self, channel):
        super().__init__()
        self.channel = channel

    def power_normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize transmitted symbols to unit average power per sample."""
        # z: [B, ...] — normalize per sample
        z_flat = z.view(z.shape[0], -1)
        power = torch.mean(z_flat ** 2, dim=1, keepdim=True)
        z_flat = z_flat / (power.sqrt() + 1e-8)
        return z_flat.view_as(z)

    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        z = self.encode(x)
        z = self.power_normalize(z)
        z = self.channel(z, snr_db)
        x_hat = self.decode(z)
        return x_hat
