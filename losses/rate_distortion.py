"""Rate-distortion loss for JSCC models."""

import torch
import torch.nn as nn


class RateDistortionLoss(nn.Module):
    """Loss = distortion + lambda * rate_estimate.

    Used by NTSCC and similar models with learned entropy models.
    """

    def __init__(self, lmbda: float = 0.01, distortion: str = "mse"):
        super().__init__()
        self.lmbda = lmbda
        if distortion == "mse":
            self.distortion_fn = nn.MSELoss()
        elif distortion == "l1":
            self.distortion_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown distortion: {distortion}")

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor,
                rate: torch.Tensor = None):
        """Compute rate-distortion loss.

        Args:
            x_hat: Reconstructed signal.
            x: Original signal.
            rate: Estimated bit-rate (scalar or per-sample).
        """
        distortion = self.distortion_fn(x_hat, x)
        if rate is not None:
            return distortion + self.lmbda * rate.mean(), {
                "distortion": distortion.item(),
                "rate": rate.mean().item(),
            }
        return distortion, {"distortion": distortion.item()}
