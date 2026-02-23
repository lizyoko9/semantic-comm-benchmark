"""ADJSCC: Attention-based Deep Joint Source-Channel Coding.

Based on: "Adaptive Deep Joint Source-Channel Coding with Attention Modules"
(Xu et al., IEEE JSAC 2021).

Key additions over DeepJSCC:
- GDN/IGDN activations instead of PReLU
- AF (Attention Feature) modules conditioned on SNR for adaptive coding
- Single model works across all SNR levels
"""

import torch
import torch.nn as nn
from models.base_model import BaseSemanticModel
from models.layers.gdn import GDN, IGDN


class AFModule(nn.Module):
    """Attention Feature module (squeeze-excitation conditioned on SNR).

    Generates channel-wise attention weights based on both
    feature statistics and the current SNR.
    """

    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(num_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(num_channels + 1, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply SNR-conditioned channel attention.

        Args:
            x: [B, C, H, W] feature map.
            snr_db: Current SNR in dB.
        """
        B, C, H, W = x.shape
        # Global average pooling
        squeeze = x.mean(dim=[2, 3])  # [B, C]
        # Append normalized SNR
        snr_feat = torch.full((B, 1), snr_db / 20.0, device=x.device)
        combined = torch.cat([squeeze, snr_feat], dim=1)  # [B, C+1]
        attn = self.fc(combined).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * attn


class ADJSCCEncoder(nn.Module):
    """ADJSCC encoder with GDN and AF modules."""

    def __init__(self, c_out: int = 16):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.gdn1 = GDN(16)
        self.af1 = AFModule(16)

        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn2 = GDN(32)
        self.af2 = AFModule(32)

        self.conv3 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.gdn3 = GDN(32)
        self.af3 = AFModule(32)

        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.gdn4 = GDN(32)
        self.af4 = AFModule(32)

        self.conv5 = nn.Conv2d(32, c_out, 5, stride=1, padding=2)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        x = self.af1(self.gdn1(self.conv1(x)), snr_db)
        x = self.af2(self.gdn2(self.conv2(x)), snr_db)
        x = self.af3(self.gdn3(self.conv3(x)), snr_db)
        x = self.af4(self.gdn4(self.conv4(x)), snr_db)
        x = self.conv5(x)
        return x


class ADJSCCDecoder(nn.Module):
    """ADJSCC decoder with IGDN and AF modules."""

    def __init__(self, c_out: int = 16):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(c_out, 32, 5, stride=1, padding=2)
        self.igdn1 = IGDN(32)
        self.af1 = AFModule(32)

        self.conv2 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
        self.igdn2 = IGDN(32)
        self.af2 = AFModule(32)

        self.conv3 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=2)
        self.igdn3 = IGDN(32)
        self.af3 = AFModule(32)

        self.conv4 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2,
                                         output_padding=1)
        self.igdn4 = IGDN(16)
        self.af4 = AFModule(16)

        self.conv5 = nn.ConvTranspose2d(16, 3, 5, stride=2, padding=2,
                                         output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor, snr_db: float) -> torch.Tensor:
        z = self.af1(self.igdn1(self.conv1(z)), snr_db)
        z = self.af2(self.igdn2(self.conv2(z)), snr_db)
        z = self.af3(self.igdn3(self.conv3(z)), snr_db)
        z = self.af4(self.igdn4(self.conv4(z)), snr_db)
        z = self.sigmoid(self.conv5(z))
        return z


class ADJSCC(BaseSemanticModel):
    """ADJSCC: SNR-adaptive joint source-channel coding.

    Args:
        channel: Channel module.
        c_out: Output channels (controls bandwidth ratio).
    """

    def __init__(self, channel, c_out: int = 16):
        super().__init__(channel)
        self.enc = ADJSCCEncoder(c_out)
        self.dec = ADJSCCDecoder(c_out)

    def encode(self, x: torch.Tensor, snr_db: float = 10.0, **kwargs) -> torch.Tensor:
        return self.enc(x, snr_db)

    def decode(self, z: torch.Tensor, snr_db: float = 10.0, **kwargs) -> torch.Tensor:
        return self.dec(z, snr_db)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        z = self.encode(x, snr_db)
        z = self.power_normalize(z)
        z = self.channel(z, snr_db)
        x_hat = self.decode(z, snr_db)
        return x_hat
