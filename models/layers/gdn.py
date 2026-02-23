"""GDN and IGDN layers (Generalized Divisive Normalization).

Based on: "Density Modeling of Images Using a Generalized Normalization
Transformation" (Ballé et al., ICLR 2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GDN(nn.Module):
    """Generalized Divisive Normalization.

    y_i = x_i / sqrt(beta_i + sum_j(gamma_ij * x_j^2))
    """

    def __init__(self, num_channels: int, inverse: bool = False):
        super().__init__()
        self.inverse = inverse
        self.beta = nn.Parameter(torch.ones(num_channels))
        self.gamma = nn.Parameter(0.1 * torch.eye(num_channels))
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        gamma = self.gamma.abs()  # Ensure non-negative
        beta = self.beta.abs() + 1e-6

        # Compute normalization factor
        x_sq = x ** 2
        # sum_j gamma_ij * x_j^2 for each channel i
        x_sq_flat = x_sq.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        norm = x_sq_flat @ gamma.t()  # [B*H*W, C]
        norm = norm.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        norm = norm + beta.view(1, -1, 1, 1)
        norm = torch.sqrt(norm)

        if self.inverse:
            return x * norm
        else:
            return x / norm


class IGDN(GDN):
    """Inverse GDN."""

    def __init__(self, num_channels: int):
        super().__init__(num_channels, inverse=True)
