"""NTSCC: Nonlinear Transform Source-Channel Coding.

Based on: "Nonlinear Transform Source-Channel Coding for Semantic Communications"
(Dai et al., IEEE JSAC 2022).

Architecture:
- Analysis transform g_a: 4×(Conv2d + GDN) → latent
- Hyper-analysis/synthesis for entropy model (side information)
- Channel encoder/decoder: Linear layers with power normalization
- Synthesis transform g_s: 4×(ConvTranspose2d + IGDN)
- Loss: MSE + λ × rate_estimate
"""

import torch
import torch.nn as nn
from models.base_model import BaseSemanticModel
from models.layers.gdn import GDN, IGDN


class AnalysisTransform(nn.Module):
    """Analysis transform g_a: image → latent representation.

    Input: [B, 3, 32, 32]
    Output: [B, M, 2, 2] (4 downsampling stages of 2x each)
    """

    def __init__(self, M: int = 128):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(3, M // 2, 5, stride=2, padding=2),    # 32→16
            GDN(M // 2),
            nn.Conv2d(M // 2, M, 5, stride=2, padding=2),    # 16→8
            GDN(M),
            nn.Conv2d(M, M, 5, stride=2, padding=2),          # 8→4
            GDN(M),
            nn.Conv2d(M, M, 5, stride=2, padding=2),          # 4→2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class SynthesisTransform(nn.Module):
    """Synthesis transform g_s: latent → reconstructed image.

    Input: [B, M, 2, 2]
    Output: [B, 3, 32, 32]
    """

    def __init__(self, M: int = 128):
        super().__init__()
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(M, M, 5, stride=2, padding=2,
                               output_padding=1),             # 2→4
            IGDN(M),
            nn.ConvTranspose2d(M, M, 5, stride=2, padding=2,
                               output_padding=1),             # 4→8
            IGDN(M),
            nn.ConvTranspose2d(M, M // 2, 5, stride=2, padding=2,
                               output_padding=1),             # 8→16
            IGDN(M // 2),
            nn.ConvTranspose2d(M // 2, 3, 5, stride=2, padding=2,
                               output_padding=1),             # 16→32
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class HyperAnalysis(nn.Module):
    """Hyper-analysis transform for entropy estimation.

    Input: [B, M, 2, 2]
    Output: [B, M//2, 1, 1]
    """

    def __init__(self, M: int = 128):
        super().__init__()
        N = M // 2
        self.transform = nn.Sequential(
            nn.Conv2d(M, N, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, 3, stride=2, padding=1),  # 2→1
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class HyperSynthesis(nn.Module):
    """Hyper-synthesis transform.

    Input: [B, M//2, 1, 1]
    Output: [B, M, 2, 2] (scale parameters for entropy model)
    """

    def __init__(self, M: int = 128):
        super().__init__()
        N = M // 2
        self.transform = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, stride=2, padding=1,
                               output_padding=1),  # 1→2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, M, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class NTSCC(BaseSemanticModel):
    """NTSCC: Nonlinear Transform Source-Channel Coding.

    Args:
        channel: Channel module.
        M: Latent dimension (channels in analysis transform output).
        k: Channel coding dimension (symbols per latent block).
        lmbda: Rate-distortion tradeoff parameter.
    """

    def __init__(self, channel, M: int = 128, k: int = 64, lmbda: float = 0.01):
        super().__init__(channel)
        self.M = M
        self.k = k
        self.lmbda = lmbda

        # Main transforms
        self.g_a = AnalysisTransform(M)
        self.g_s = SynthesisTransform(M)

        # Hyper transforms for rate estimation
        self.h_a = HyperAnalysis(M)
        self.h_s = HyperSynthesis(M)

        # Channel encoder/decoder: latent → channel symbols
        latent_dim = M * 2 * 2  # M channels × 2×2 spatial
        self.channel_encoder = nn.Sequential(
            nn.Linear(latent_dim, k * 2),
            nn.PReLU(k * 2),
            nn.Linear(k * 2, k * 2),
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(k * 2, k * 2),
            nn.PReLU(k * 2),
            nn.Linear(k * 2, latent_dim),
        )

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y = self.g_a(x)                          # [B, M, 2, 2]
        B = y.shape[0]
        y_flat = y.view(B, -1)                   # [B, M*4]
        z = self.channel_encoder(y_flat)          # [B, k*2]
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        y_hat_flat = self.channel_decoder(z)      # [B, M*4]
        B = y_hat_flat.shape[0]
        y_hat = y_hat_flat.view(B, self.M, 2, 2)
        x_hat = self.g_s(y_hat)                   # [B, 3, 32, 32]
        return x_hat

    def forward(self, x: torch.Tensor, snr_db: float):
        """Forward pass with rate estimation.

        Returns:
            x_hat: Reconstructed image.
            rate: Estimated rate (for rate-distortion loss).
        """
        # Analysis
        y = self.g_a(x)                           # [B, M, 2, 2]

        # Hyper-analysis for rate estimation
        z_hyper = self.h_a(y)                     # [B, M//2, 1, 1]
        sigma = self.h_s(z_hyper)                 # [B, M, 2, 2]
        # Rate estimate: log of scale parameters
        rate = torch.log(sigma.abs() + 1e-6).sum(dim=[1, 2, 3])  # [B]

        # Channel coding
        B = y.shape[0]
        y_flat = y.view(B, -1)
        z = self.channel_encoder(y_flat)
        z = self.power_normalize(z)

        # Channel transmission
        z = self.channel(z, snr_db)

        # Channel decoding
        y_hat_flat = self.channel_decoder(z)
        y_hat = y_hat_flat.view(B, self.M, 2, 2)

        # Synthesis
        x_hat = self.g_s(y_hat)

        return x_hat, rate
