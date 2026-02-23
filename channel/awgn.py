"""Additive White Gaussian Noise (AWGN) channel."""

import torch
from channel.base_channel import BaseChannel


class AWGNChannel(BaseChannel):
    """AWGN channel: y = x + n, where n ~ N(0, sigma^2)."""

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        if not self.training and snr_db is None:
            return x
        noise_std = self.snr_to_noise_std(snr_db)
        noise = torch.randn_like(x) * noise_std
        return x + noise
