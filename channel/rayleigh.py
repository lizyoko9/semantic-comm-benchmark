"""Rayleigh fading channel."""

import torch
from channel.base_channel import BaseChannel


class RayleighChannel(BaseChannel):
    """Rayleigh fading channel: y = h*x + n.

    h ~ CN(0,1) → |h| is Rayleigh distributed.
    Represented as real: h_real, h_imag ~ N(0, 0.5) each.
    For real-valued signals, we apply |h| * x + n with perfect CSI equalization.
    """

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        noise_std = self.snr_to_noise_std(snr_db)

        # Generate complex fading coefficients
        h_real = torch.randn(x.shape[0], 1, device=x.device) * (0.5 ** 0.5)
        h_imag = torch.randn(x.shape[0], 1, device=x.device) * (0.5 ** 0.5)
        h_mag = (h_real ** 2 + h_imag ** 2).sqrt()

        # Reshape for broadcasting
        extra_dims = [1] * (x.dim() - 1)
        h_mag = h_mag.view(x.shape[0], *extra_dims)

        # Fading + noise, then perfect CSI equalization
        noise = torch.randn_like(x) * noise_std
        y = h_mag * x + noise
        # Equalize: divide by h_mag (zero-forcing)
        y = y / (h_mag + 1e-8)
        return y
