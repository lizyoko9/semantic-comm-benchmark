"""DeepJSCC: Deep Joint Source-Channel Coding for image transmission.

Based on: "Deep Joint Source-Channel Coding for Wireless Image Transmission"
(Bourtsoulatze et al., IEEE TCCN 2019).

Architecture: 5-layer CNN autoencoder with PReLU activations.
Encoder: Conv2d layers with stride-2 downsampling for first two layers.
Decoder: ConvTranspose2d layers mirroring the encoder.
"""

import torch
import torch.nn as nn
from models.base_model import BaseSemanticModel


class DeepJSCCEncoder(nn.Module):
    """5-layer CNN encoder.

    Input: [B, 3, 32, 32]
    Output: [B, C_out, 8, 8]
    """

    def __init__(self, c_out: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 3 → 16, stride 2: 32→16
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(16),
            # Layer 2: 16 → 32, stride 2: 16→8
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(32),
            # Layer 3: 32 → 32, stride 1: 8→8
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            # Layer 4: 32 → 32, stride 1: 8→8
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            # Layer 5: 32 → c_out, stride 1: 8→8
            nn.Conv2d(32, c_out, kernel_size=5, stride=1, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeepJSCCDecoder(nn.Module):
    """5-layer CNN decoder (mirror of encoder).

    Input: [B, C_out, 8, 8]
    Output: [B, 3, 32, 32]
    """

    def __init__(self, c_out: int = 16):
        super().__init__()
        self.decoder = nn.Sequential(
            # Layer 1: c_out → 32
            nn.ConvTranspose2d(c_out, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            # Layer 2: 32 → 32
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            # Layer 3: 32 → 32
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            # Layer 4: 32 → 16, stride 2: 8→16
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.PReLU(16),
            # Layer 5: 16 → 3, stride 2: 16→32
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class DeepJSCC(BaseSemanticModel):
    """DeepJSCC model for CIFAR-10 (32x32).

    Args:
        channel: Channel module.
        c_out: Number of output channels (controls bandwidth ratio).
               c_out=16 → ratio ≈ 16*8*8 / (3*32*32) ≈ 1/3
               c_out=8  → ratio ≈ 1/6
    """

    def __init__(self, channel, c_out: int = 16):
        super().__init__(channel)
        self.encoder = DeepJSCCEncoder(c_out)
        self.decoder = DeepJSCCDecoder(c_out)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decoder(z)
