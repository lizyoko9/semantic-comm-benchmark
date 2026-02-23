"""Metrics factory."""

from metrics.psnr_ssim import compute_psnr, compute_ssim
from metrics.lpips_metric import LPIPSMetric
from metrics.bleu_similarity import compute_bleu, compute_sentence_similarity


def get_metric(name: str, **kwargs):
    """Get a metric function or object by name."""
    name = name.lower()
    if name == "psnr":
        return compute_psnr
    elif name == "ssim":
        return compute_ssim
    elif name == "lpips":
        return LPIPSMetric(**kwargs)
    elif name == "bleu":
        return compute_bleu
    elif name == "sentence_similarity":
        return compute_sentence_similarity
    else:
        raise ValueError(f"Unknown metric: {name}. "
                         "Available: psnr, ssim, lpips, bleu, sentence_similarity")
