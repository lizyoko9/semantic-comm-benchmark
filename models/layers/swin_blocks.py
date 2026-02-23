"""Swin Transformer blocks for WITT model.

Simplified implementation of Swin Transformer blocks with
window-based multi-head self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij')).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: [B, H, W, C]
        window_size: window size

    Returns:
        windows: [B * num_windows, window_size * window_size, C]
    """
    B, H, W, C = x.shape
    nH = H // window_size
    nW = W // window_size
    x = x.view(B, nH, window_size, nW, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B * nH * nW, window_size * window_size, C)
    return x


def window_reverse(windows: torch.Tensor, window_size: int,
                    H: int, W: int) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: [B * num_windows, window_size * window_size, C]

    Returns:
        x: [B, H, W, C]
    """
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, H, W, C]
        """
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)
        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]

        # Window partition
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, patch_size: int = 2, in_channels: int = 3,
                 embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, H//p, W//p, embed_dim]"""
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.permute(0, 2, 3, 1)  # [B, H', W', embed_dim]
        return x


class PatchExpand(nn.Module):
    """Patch expanding (upsample) layer."""

    def __init__(self, dim: int, out_dim: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.expand = nn.Linear(dim, out_dim * scale * scale)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, W, C] → [B, H*s, W*s, out_dim]"""
        B, H, W, C = x.shape
        x = self.expand(x)  # [B, H, W, out_dim * s^2]
        x = x.view(B, H, W, self.scale, self.scale, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.scale, W * self.scale, self.out_dim)
        return x


class Downsample(nn.Module):
    """Spatial downsampling via linear projection of 2x2 patches."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, out_dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, W, C] → [B, H//2, W//2, out_dim]"""
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Upsample(nn.Module):
    """Spatial upsampling via linear expansion."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.expand = PatchExpand(dim, out_dim, scale=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.norm(x)
        return x
