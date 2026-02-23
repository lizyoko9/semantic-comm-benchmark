"""Checkpoint save/load utilities."""

import os
import torch


def save_checkpoint(model, optimizer, epoch: int, metrics: dict,
                    filepath: str, **extra):
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    state.update(extra)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model, optimizer=None, device="cpu"):
    """Load model checkpoint. Returns (epoch, metrics)."""
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})
