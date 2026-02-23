#!/usr/bin/env python3
"""Train a semantic communication model.

Usage:
    python scripts/train.py --config configs/deepjscc.yaml
    python scripts/train.py --config configs/adjscc.yaml --device cuda:0
    python scripts/train.py --config configs/deepjscc.yaml --resume checkpoints/deepjscc/best.pt
"""

import argparse
import sys
import os
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from data import get_dataloader
from trainer.image_trainer import ImageTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config, merged with defaults."""
    # Load defaults
    default_path = os.path.join(os.path.dirname(config_path), "default.yaml")
    config = {}
    if os.path.exists(default_path):
        with open(default_path) as f:
            config = yaml.safe_load(f)

    # Override with model config
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    config.update(model_config)

    return config


def main():
    parser = argparse.ArgumentParser(description="Train semantic comm model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, cpu, mps)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create model
    model_cfg = config.get("model", {})
    model_name = model_cfg.pop("name", config.get("model_name", "deepjscc"))
    channel_type = config.get("channel_type", "awgn")
    model = get_model(model_name, channel_type=channel_type, **model_cfg)
    print(f"Model: {model_name} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    dataset = config.get("dataset", "cifar10")
    train_loader, test_loader = get_dataloader(
        dataset, split="both",
        batch_size=config.get("batch_size", 128),
        data_root=config.get("data_root", "./data_cache"),
        num_workers=config.get("num_workers", 4),
    )

    # Create trainer
    config["model_name"] = model_name
    trainer = ImageTrainer(model, train_loader, test_loader, config, device=device)

    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
