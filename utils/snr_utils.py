"""SNR utility functions for semantic communication."""

import torch
import math


def db_to_linear(snr_db: float) -> float:
    """Convert SNR from dB to linear scale."""
    return 10.0 ** (snr_db / 10.0)


def linear_to_db(snr_linear: float) -> float:
    """Convert SNR from linear scale to dB."""
    return 10.0 * math.log10(snr_linear)


def snr_to_noise_std(snr_db: float, signal_power: float = 1.0) -> float:
    """Compute noise standard deviation from SNR in dB.

    Assumes complex channel where noise power = signal_power / snr_linear.
    Returns std = sqrt(noise_power / 2) per real dimension.
    """
    snr_linear = db_to_linear(snr_db)
    noise_power = signal_power / snr_linear
    return math.sqrt(noise_power / 2.0)


def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Empirically compute SNR in dB from signal and noise tensors."""
    signal_power = torch.mean(signal ** 2).item()
    noise_power = torch.mean(noise ** 2).item()
    if noise_power == 0:
        return float('inf')
    return 10.0 * math.log10(signal_power / noise_power)
