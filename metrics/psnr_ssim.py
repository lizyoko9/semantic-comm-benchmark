"""PSNR and SSIM metrics for image quality evaluation."""

import torch
import torch.nn.functional as F
import math


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        img1, img2: [B, C, H, W] tensors in [0, max_val].
        max_val: Maximum pixel value.

    Returns:
        Average PSNR in dB.
    """
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(max_val ** 2 / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11, max_val: float = 1.0) -> float:
    """Compute Structural Similarity Index.

    Uses a Gaussian window. Operates on each channel independently.

    Args:
        img1, img2: [B, C, H, W] tensors in [0, max_val].

    Returns:
        Average SSIM.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    channels = img1.shape[1]
    window = window.expand(channels, 1, -1, -1).contiguous()

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()
