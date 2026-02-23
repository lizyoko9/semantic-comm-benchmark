#!/usr/bin/env python3
"""Evaluate a trained model at multiple SNR points.

Usage:
    python scripts/evaluate.py --config configs/deepjscc.yaml \
        --checkpoint checkpoints/deepjscc/best.pt
    python scripts/evaluate.py --config configs/adjscc.yaml \
        --checkpoint checkpoints/adjscc/best.pt --snr -2 0 4 8 12 16 20
"""

import argparse
import sys
import os
import yaml
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from data import get_dataloader
from trainer.image_trainer import ImageTrainer
from utils.checkpoint import load_checkpoint


def load_config(config_path: str) -> dict:
    default_path = os.path.join(os.path.dirname(config_path), "default.yaml")
    config = {}
    if os.path.exists(default_path):
        with open(default_path) as f:
            config = yaml.safe_load(f)
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    config.update(model_config)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic comm model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--snr", type=float, nargs="+", default=None,
                        help="SNR points to evaluate at")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Create model
    model_cfg = config.get("model", {})
    model_name = model_cfg.pop("name", config.get("model_name", "deepjscc"))
    channel_type = config.get("channel_type", "awgn")
    model = get_model(model_name, channel_type=channel_type, **model_cfg)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} from {args.checkpoint}")

    # Load data
    dataset = config.get("dataset", "cifar10")
    _, test_loader = get_dataloader(
        dataset, split="both",
        batch_size=config.get("batch_size", 128),
        data_root=config.get("data_root", "./data_cache"),
        num_workers=config.get("num_workers", 4),
    )

    # Evaluate
    snr_list = args.snr or config.get("snr_range",
                                       [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    config["model_name"] = model_name
    trainer = ImageTrainer(model, None, test_loader, config, device=device)
    results = trainer.evaluate_snr_sweep(snr_list)

    # Print results
    print(f"\n{'SNR (dB)':>10} {'PSNR (dB)':>10} {'SSIM':>10}")
    print("-" * 35)
    for snr in sorted(results.keys()):
        m = results[snr]
        print(f"{snr:>10.1f} {m['psnr']:>10.2f} {m['ssim']:>10.4f}")

    # Save results
    if args.output:
        # Convert keys to strings for JSON
        json_results = {str(k): v for k, v in results.items()}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"model": model_name, "channel": channel_type,
                        "results": json_results}, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
