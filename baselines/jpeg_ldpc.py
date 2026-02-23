"""JPEG + simulated LDPC/BPSK baseline.

Traditional separate source-channel coding baseline:
1. JPEG compression at given quality
2. BPSK modulation
3. Simulated BER based on SNR (AWGN/Rayleigh)
4. Bit error injection into compressed bitstream
5. JPEG decompression
"""

import io
import math
import numpy as np
from PIL import Image
import torch
from scipy.special import erfc


def bpsk_ber_awgn(snr_db: float) -> float:
    """Theoretical BER for uncoded BPSK over AWGN channel."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 0.5 * erfc(math.sqrt(snr_lin))


def ldpc_ber_awgn(snr_db: float, code_rate: float = 0.5,
                   coding_gain_db: float = 6.0) -> float:
    """Approximate BER for LDPC-coded BPSK over AWGN.

    Approximation: LDPC provides ~6dB coding gain at moderate BER.
    """
    effective_snr = snr_db + coding_gain_db
    # Account for rate loss
    effective_snr -= 10 * math.log10(1.0 / code_rate)
    return bpsk_ber_awgn(effective_snr)


def inject_bit_errors(data: bytes, ber: float) -> bytes:
    """Inject random bit errors into byte array at given BER."""
    if ber <= 0 or ber >= 1:
        if ber <= 0:
            return data
        return bytes(np.random.randint(0, 256, len(data), dtype=np.uint8))

    arr = np.frombuffer(data, dtype=np.uint8).copy()
    # Generate bit flip mask
    bits = np.unpackbits(arr)
    mask = np.random.random(len(bits)) < ber
    bits ^= mask.astype(np.uint8)
    arr = np.packbits(bits)
    return bytes(arr[:len(data)])


def jpeg_encode(img_tensor: torch.Tensor, quality: int = 50) -> bytes:
    """JPEG encode a single image tensor [C, H, W] in [0, 1]."""
    img = img_tensor.detach().cpu().clamp(0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_decode(data: bytes, shape: tuple) -> torch.Tensor:
    """JPEG decode bytes back to tensor [C, H, W] in [0, 1]."""
    try:
        buf = io.BytesIO(data)
        pil_img = Image.open(buf).convert("RGB")
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)
    except Exception:
        # If JPEG decode fails (too many bit errors), return grey image
        return torch.full((shape[0], shape[1], shape[2]), 0.5)


class JPEGLDPCBaseline:
    """JPEG + LDPC/BPSK baseline for image transmission.

    Args:
        quality: JPEG compression quality (1-100).
        code_rate: LDPC code rate.
        coding_gain_db: Approximate LDPC coding gain in dB.
    """

    def __init__(self, quality: int = 50, code_rate: float = 0.5,
                 coding_gain_db: float = 6.0):
        self.quality = quality
        self.code_rate = code_rate
        self.coding_gain_db = coding_gain_db

    def __call__(self, images: torch.Tensor, snr_db: float,
                 channel_type: str = "awgn") -> torch.Tensor:
        """Process batch of images through JPEG + LDPC pipeline.

        Args:
            images: [B, C, H, W] in [0, 1].
            snr_db: Channel SNR in dB.
            channel_type: "awgn" or "rayleigh".

        Returns:
            Reconstructed images [B, C, H, W].
        """
        # Compute BER
        ber = ldpc_ber_awgn(snr_db, self.code_rate, self.coding_gain_db)
        if channel_type == "rayleigh":
            # Rayleigh roughly doubles the required SNR
            ber = ldpc_ber_awgn(snr_db - 3.0, self.code_rate,
                                self.coding_gain_db)

        B, C, H, W = images.shape
        results = []

        for i in range(B):
            # Compress
            compressed = jpeg_encode(images[i], self.quality)
            # Simulate channel errors
            corrupted = inject_bit_errors(compressed, ber)
            # Decompress
            recon = jpeg_decode(corrupted, (C, H, W))
            results.append(recon)

        return torch.stack(results).to(images.device)

    def bandwidth_ratio(self, images: torch.Tensor) -> float:
        """Compute average bandwidth ratio (compressed bits / source bits)."""
        total_compressed = 0
        B, C, H, W = images.shape
        source_bits = C * H * W * 8  # 8 bits per pixel

        for i in range(min(B, 10)):  # Sample first 10 images
            compressed = jpeg_encode(images[i], self.quality)
            total_compressed += len(compressed) * 8 / self.code_rate

        avg_channel_bits = total_compressed / min(B, 10)
        return avg_channel_bits / source_bits
