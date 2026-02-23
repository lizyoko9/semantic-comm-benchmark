"""LPIPS perceptual loss wrapper."""

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss."""

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net=net)
        # Freeze LPIPS network
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        """Compute LPIPS perceptual loss.

        Inputs should be in [0, 1] range; LPIPS expects [-1, 1].
        """
        # Scale to [-1, 1]
        x_scaled = 2.0 * x - 1.0
        x_hat_scaled = 2.0 * x_hat - 1.0
        return self.lpips_fn(x_hat_scaled, x_scaled).mean()
