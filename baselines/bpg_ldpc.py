"""BPG + simulated LDPC/BPSK baseline.

BPG (Better Portable Graphics) uses HEVC intra coding.
Since BPG requires external tools, this implementation uses Pillow's
WebP as a higher-efficiency substitute when BPG is not available,
or calls the bpgenc/bpgdec binaries if found on the system.
"""

import io
import os
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch

from baselines.jpeg_ldpc import ldpc_ber_awgn, inject_bit_errors


def _bpg_available() -> bool:
    """Check if bpgenc and bpgdec are available."""
    try:
        subprocess.run(["bpgenc", "--help"], capture_output=True, timeout=5)
        subprocess.run(["bpgdec", "--help"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def bpg_encode(img_tensor: torch.Tensor, quality: int = 30) -> bytes:
    """Encode image using BPG (or WebP fallback).

    Args:
        img_tensor: [C, H, W] in [0, 1].
        quality: Compression quality.
    """
    img_np = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    if _bpg_available():
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_in:
            pil_img.save(f_in.name)
            f_out = f_in.name.replace(".png", ".bpg")
            subprocess.run(
                ["bpgenc", "-q", str(quality), "-o", f_out, f_in.name],
                capture_output=True, timeout=30,
            )
            with open(f_out, "rb") as f:
                data = f.read()
            os.unlink(f_in.name)
            os.unlink(f_out)
            return data
    else:
        # Fallback to WebP (better compression than JPEG)
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=quality)
        return buf.getvalue()


def bpg_decode(data: bytes, shape: tuple) -> torch.Tensor:
    """Decode BPG (or WebP fallback) data."""
    try:
        if _bpg_available() and len(data) > 4 and data[:4] == b"BPG\xfb":
            with tempfile.NamedTemporaryFile(suffix=".bpg", delete=False) as f_in:
                f_in.write(data)
                f_out = f_in.name.replace(".bpg", ".png")
                subprocess.run(
                    ["bpgdec", "-o", f_out, f_in.name],
                    capture_output=True, timeout=30,
                )
                pil_img = Image.open(f_out).convert("RGB")
                os.unlink(f_in.name)
                os.unlink(f_out)
        else:
            buf = io.BytesIO(data)
            pil_img = Image.open(buf).convert("RGB")

        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)
    except Exception:
        return torch.full((shape[0], shape[1], shape[2]), 0.5)


class BPGLDPCBaseline:
    """BPG + LDPC/BPSK baseline (or WebP + LDPC when BPG unavailable).

    Args:
        quality: Compression quality.
        code_rate: LDPC code rate.
        coding_gain_db: Approximate LDPC coding gain.
    """

    def __init__(self, quality: int = 30, code_rate: float = 0.5,
                 coding_gain_db: float = 6.0):
        self.quality = quality
        self.code_rate = code_rate
        self.coding_gain_db = coding_gain_db
        self.uses_bpg = _bpg_available()

    def __call__(self, images: torch.Tensor, snr_db: float,
                 channel_type: str = "awgn") -> torch.Tensor:
        """Process batch through BPG/WebP + LDPC pipeline."""
        ber = ldpc_ber_awgn(snr_db, self.code_rate, self.coding_gain_db)
        if channel_type == "rayleigh":
            ber = ldpc_ber_awgn(snr_db - 3.0, self.code_rate,
                                self.coding_gain_db)

        B, C, H, W = images.shape
        results = []

        for i in range(B):
            compressed = bpg_encode(images[i], self.quality)
            corrupted = inject_bit_errors(compressed, ber)
            recon = bpg_decode(corrupted, (C, H, W))
            results.append(recon)

        return torch.stack(results).to(images.device)

    @property
    def codec_name(self) -> str:
        return "BPG" if self.uses_bpg else "WebP"
