"""LPIPS metric for perceptual quality evaluation."""

import torch


class LPIPSMetric:
    """LPIPS evaluation metric (lower is better)."""

    def __init__(self, net: str = "alex", device: str = "cpu"):
        import lpips
        self.fn = lpips.LPIPS(net=net).to(device)
        self.fn.eval()

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute LPIPS distance.

        Args:
            img1, img2: [B, C, H, W] in [0, 1].
        """
        x1 = 2.0 * img1 - 1.0
        x2 = 2.0 * img2 - 1.0
        return self.fn(x1, x2).mean().item()
