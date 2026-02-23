"""WITT: Wireless Image Transmission Transformer.

Based on: "WITT: A Wireless Image Transmission Transformer for Semantic
Communications" (Yang et al., IEEE JSAC 2023).

Architecture:
- Patch embedding (patch_size=2)
- 2-stage Swin Transformer encoder with downsampling
- Channel bottleneck with ModNet (SNR-conditioned modulation)
- 2-stage Swin Transformer decoder with upsampling
- Patch reconstruction
"""

import torch
import torch.nn as nn
from models.base_model import BaseSemanticModel
from models.layers.swin_blocks import (
    SwinTransformerBlock, PatchEmbed, PatchExpand,
    Downsample, Upsample,
)


class ModNet(nn.Module):
    """Modulation network: SNR → affine parameters for feature modulation.

    Generates per-channel scale and bias conditioned on SNR.
    """

    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Linear(hidden_dim, dim)
        self.bias = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Apply SNR-conditioned affine modulation.

        Args:
            x: [B, H, W, C] or [B, N, C] features.
            snr_db: Current SNR in dB.
        """
        B = x.shape[0]
        snr_feat = torch.tensor([[snr_db / 20.0]], device=x.device).expand(B, 1)
        h = self.net(snr_feat)
        scale = self.scale(h) + 1.0  # Initialize around 1
        bias = self.bias(h)

        if x.dim() == 4:  # [B, H, W, C]
            scale = scale.unsqueeze(1).unsqueeze(1)
            bias = bias.unsqueeze(1).unsqueeze(1)
        elif x.dim() == 3:  # [B, N, C]
            scale = scale.unsqueeze(1)
            bias = bias.unsqueeze(1)

        return x * scale + bias


class SwinEncoderStage(nn.Module):
    """One stage of Swin Transformer encoder."""

    def __init__(self, dim: int, num_heads: int, depth: int = 2,
                 window_size: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SwinDecoderStage(nn.Module):
    """One stage of Swin Transformer decoder."""

    def __init__(self, dim: int, num_heads: int, depth: int = 2,
                 window_size: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class WITT(BaseSemanticModel):
    """WITT: Wireless Image Transmission Transformer.

    For 32×32 CIFAR-10:
    - Patch embed (p=2): 32→16×16, dim=128
    - Stage 1 encoder: 16×16, dim=128, heads=4
    - Downsample: 16→8×8, dim=256
    - Stage 2 encoder: 8×8, dim=256, heads=8
    - Bottleneck: flatten + linear → channel symbols
    - Stage 2 decoder + upsample + Stage 1 decoder
    - Patch expand to reconstruct image

    Args:
        channel: Channel module.
        patch_size: Patch embedding size.
        embed_dim: Stage 1 embedding dimension.
        channel_dim: Number of channel symbols (bottleneck size).
        depths: Number of Swin blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Window size for attention.
    """

    def __init__(self, channel, patch_size: int = 2, embed_dim: int = 128,
                 channel_dim: int = 256, depths=(2, 2), num_heads=(4, 8),
                 window_size: int = 8):
        super().__init__(channel)
        self.embed_dim = embed_dim
        self.channel_dim = channel_dim
        dim2 = embed_dim * 2  # Stage 2 dimension

        # Encoder
        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.enc_stage1 = SwinEncoderStage(embed_dim, num_heads[0], depths[0],
                                            window_size)
        self.downsample = Downsample(embed_dim, dim2)
        self.enc_stage2 = SwinEncoderStage(dim2, num_heads[1], depths[1],
                                            window_size)

        # Bottleneck: [B, 8*8, dim2] → [B, channel_dim]
        # For 32x32 with patch=2 and one downsample: H'=8, W'=8
        self.bottleneck_h = 8
        self.bottleneck_w = 8
        bottleneck_flat = self.bottleneck_h * self.bottleneck_w * dim2
        self.channel_enc = nn.Linear(bottleneck_flat, channel_dim)
        self.channel_dec = nn.Linear(channel_dim, bottleneck_flat)

        # SNR modulation
        self.mod_enc = ModNet(dim2)
        self.mod_dec = ModNet(dim2)

        # Decoder
        self.dec_stage2 = SwinDecoderStage(dim2, num_heads[1], depths[1],
                                            window_size)
        self.upsample = Upsample(dim2, embed_dim)
        self.dec_stage1 = SwinDecoderStage(embed_dim, num_heads[0], depths[0],
                                            window_size)

        # Final projection: [B, 16, 16, embed_dim] → [B, 3, 32, 32]
        self.final_expand = PatchExpand(embed_dim, 3, scale=patch_size)
        self.final_proj = nn.Linear(3, 3)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: torch.Tensor, snr_db: float = 10.0, **kwargs) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)          # [B, 16, 16, 128]
        x = self.enc_stage1(x)           # [B, 16, 16, 128]
        x = self.downsample(x)           # [B, 8, 8, 256]
        x = self.enc_stage2(x)           # [B, 8, 8, 256]
        x = self.mod_enc(x, snr_db)      # SNR modulation
        x = x.reshape(B, -1)             # [B, 8*8*256]
        z = self.channel_enc(x)          # [B, channel_dim]
        return z

    def decode(self, z: torch.Tensor, snr_db: float = 10.0, **kwargs) -> torch.Tensor:
        B = z.shape[0]
        dim2 = self.embed_dim * 2
        x = self.channel_dec(z)                                  # [B, 8*8*256]
        x = x.view(B, self.bottleneck_h, self.bottleneck_w, dim2)  # [B, 8, 8, 256]
        x = self.mod_dec(x, snr_db)
        x = self.dec_stage2(x)           # [B, 8, 8, 256]
        x = self.upsample(x)            # [B, 16, 16, 128]
        x = self.dec_stage1(x)           # [B, 16, 16, 128]
        x = self.final_expand(x)         # [B, 32, 32, 3]
        x = self.final_proj(x)
        x = x.permute(0, 3, 1, 2)       # [B, 3, 32, 32]
        x = self.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        z = self.encode(x, snr_db)
        z = self.power_normalize(z)
        z = self.channel(z, snr_db)
        x_hat = self.decode(z, snr_db)
        return x_hat
