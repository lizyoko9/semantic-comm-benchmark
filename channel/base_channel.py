"""Channel model abstraction for signal transmission."""

import torch
import torch.nn as nn
from abc import abstractmethod
from utils.snr_utils import snr_to_noise_std


class BaseChannel(nn.Module):
    """Abstract base class for channel models.

    All channels expect complex-valued input represented as real tensors
    with last dimension = 2 (real, imag), or simply real-valued tensors.
    """

    def __init__(self):
        super().__init__()

    def snr_to_noise_std(self, snr_db: float, signal_power: float = 1.0) -> float:
        return snr_to_noise_std(snr_db, signal_power)

    @abstractmethod
    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Transmit signal through channel.

        Args:
            x: Input signal tensor (power-normalized).
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Received signal tensor (same shape as x).
        """
        ...
