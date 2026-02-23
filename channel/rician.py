"""Rician fading channel."""

import math
import torch
from channel.base_channel import BaseChannel


class RicianChannel(BaseChannel):
    """Rician fading channel: y = h*x + n.

    h = sqrt(K/(K+1)) * h_LoS + sqrt(1/(K+1)) * h_scatter
    where h_LoS = 1 (deterministic LoS component) and
    h_scatter ~ CN(0,1) (scattered component).
    K is the Rician K-factor (linear).
    """

    def __init__(self, k_factor_db: float = 10.0):
        super().__init__()
        k_linear = 10.0 ** (k_factor_db / 10.0)
        self.register_buffer("k_linear", torch.tensor(k_linear))

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        noise_std = self.snr_to_noise_std(snr_db)
        k = self.k_linear.item()

        # LoS component
        los_scale = math.sqrt(k / (k + 1.0))
        # Scatter component
        scatter_scale = math.sqrt(1.0 / (k + 1.0))

        h_scatter_real = torch.randn(x.shape[0], 1, device=x.device) * (0.5 ** 0.5)
        h_scatter_imag = torch.randn(x.shape[0], 1, device=x.device) * (0.5 ** 0.5)

        h_real = los_scale + scatter_scale * h_scatter_real
        h_imag = scatter_scale * h_scatter_imag
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()

        extra_dims = [1] * (x.dim() - 1)
        h_mag = h_mag.view(x.shape[0], *extra_dims)

        noise = torch.randn_like(x) * noise_std
        y = h_mag * x + noise
        y = y / (h_mag + 1e-8)
        return y
