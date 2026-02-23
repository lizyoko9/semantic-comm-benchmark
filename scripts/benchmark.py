#!/usr/bin/env python3
"""Full benchmark: all models x SNRs x channels → tables + plots.

Usage:
    python scripts/benchmark.py --models deepjscc adjscc --channels awgn \
        --snr -2 0 4 8 12 16 20 --output results/

    # Full sweep with trained checkpoints
    python scripts/benchmark.py --checkpoint_dir checkpoints/ --output results/
"""

import argparse
import sys
import os
import json
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from data import get_dataloader
from metrics.psnr_ssim import compute_psnr, compute_ssim
from baselines.jpeg_ldpc import JPEGLDPCBaseline
from baselines.bpg_ldpc import BPGLDPCBaseline
from utils.checkpoint import load_checkpoint


DEFAULT_SNRS = [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
DEFAULT_MODELS = ["deepjscc", "adjscc", "ntscc", "witt"]
DEFAULT_CHANNELS = ["awgn"]
MODEL_CONFIGS = {
    "deepjscc": "configs/deepjscc.yaml",
    "adjscc": "configs/adjscc.yaml",
    "ntscc": "configs/ntscc.yaml",
    "witt": "configs/witt.yaml",
}


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


@torch.no_grad()
def evaluate_model(model, test_loader, snr_list, device, use_rate_loss=False):
    """Evaluate a model at multiple SNR points."""
    model.eval()
    results = {}

    for snr_db in snr_list:
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0

        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            if use_rate_loss:
                x_hat, _ = model(x, snr_db)
            else:
                x_hat = model(x, snr_db)

            x_hat = x_hat.clamp(0, 1)
            total_psnr += compute_psnr(x, x_hat)
            total_ssim += compute_ssim(x, x_hat)
            n_batches += 1

        n = max(n_batches, 1)
        results[snr_db] = {"psnr": total_psnr / n, "ssim": total_ssim / n}

    return results


@torch.no_grad()
def evaluate_baseline(baseline, test_loader, snr_list, channel_type):
    """Evaluate a traditional baseline at multiple SNR points."""
    results = {}

    for snr_db in snr_list:
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0

        for batch in test_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x_hat = baseline(x, snr_db, channel_type)
            x_hat = x_hat.clamp(0, 1)
            total_psnr += compute_psnr(x, x_hat)
            total_ssim += compute_ssim(x, x_hat)
            n_batches += 1

        n = max(n_batches, 1)
        results[snr_db] = {"psnr": total_psnr / n, "ssim": total_ssim / n}

    return results


def plot_results(all_results: dict, metric: str, output_path: str,
                 channel_type: str):
    """Generate PSNR/SSIM vs SNR plot."""
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10.colors

    for i, (name, results) in enumerate(all_results.items()):
        snrs = sorted(results.keys())
        values = [results[s][metric] for s in snrs]
        plt.plot(snrs, values, marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=name, linewidth=2,
                 markersize=8)

    plt.xlabel("SNR (dB)", fontsize=14)
    ylabel = "PSNR (dB)" if metric == "psnr" else "SSIM"
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f"{ylabel} vs SNR — {channel_type.upper()} Channel", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def print_table(all_results: dict, metric: str, snr_list: list):
    """Print results table."""
    header = f"{'Model':<15}" + "".join(f"{s:>8}" for s in snr_list)
    print(f"\n{metric.upper()} Results:")
    print(header)
    print("-" * len(header))

    for name, results in all_results.items():
        row = f"{name:<15}"
        for snr in snr_list:
            if snr in results:
                val = results[snr][metric]
                row += f"{val:>8.2f}" if metric == "psnr" else f"{val:>8.4f}"
            else:
                row += f"{'N/A':>8}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Full benchmark sweep")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to evaluate")
    parser.add_argument("--channels", type=str, nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--snr", type=float, nargs="+", default=DEFAULT_SNRS)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--baselines", action="store_true", default=True,
                        help="Include JPEG/BPG baselines")
    parser.add_argument("--no-baselines", action="store_false", dest="baselines")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Determine which models to evaluate
    model_names = args.models
    if model_names is None:
        # Find available checkpoints
        model_names = []
        for name in DEFAULT_MODELS:
            ckpt = os.path.join(args.checkpoint_dir, name, "best.pt")
            if os.path.exists(ckpt):
                model_names.append(name)
        if not model_names:
            print("No trained checkpoints found. Run training first or specify --models")
            print("Available configs:", list(MODEL_CONFIGS.keys()))
            return

    # Load test data
    _, test_loader = get_dataloader(
        "cifar10", split="both", batch_size=args.batch_size,
        data_root="./data_cache", num_workers=4,
    )

    for channel_type in args.channels:
        print(f"\n{'='*60}")
        print(f"Channel: {channel_type.upper()}")
        print(f"{'='*60}")

        all_results = {}

        # Evaluate neural models
        for model_name in model_names:
            ckpt_path = os.path.join(args.checkpoint_dir, model_name, "best.pt")
            config_path = MODEL_CONFIGS.get(model_name)

            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}, skipping {model_name}")
                continue

            print(f"\nEvaluating {model_name}...")
            config = load_config(config_path) if config_path else {}
            model_cfg = config.get("model", {})
            model_cfg.pop("name", None)

            model = get_model(model_name, channel_type=channel_type, **model_cfg)
            load_checkpoint(ckpt_path, model, device=device)
            model = model.to(device)

            use_rate_loss = config.get("use_rate_loss", False)
            results = evaluate_model(model, test_loader, args.snr, device,
                                     use_rate_loss)
            all_results[model_name] = results

        # Evaluate baselines
        if args.baselines:
            print("\nEvaluating JPEG+LDPC baseline...")
            jpeg_baseline = JPEGLDPCBaseline(quality=50)
            all_results["JPEG+LDPC"] = evaluate_baseline(
                jpeg_baseline, test_loader, args.snr, channel_type)

            print("Evaluating BPG/WebP+LDPC baseline...")
            bpg_baseline = BPGLDPCBaseline(quality=30)
            baseline_name = f"{bpg_baseline.codec_name}+LDPC"
            all_results[baseline_name] = evaluate_baseline(
                bpg_baseline, test_loader, args.snr, channel_type)

        # Print tables
        print_table(all_results, "psnr", args.snr)
        print_table(all_results, "ssim", args.snr)

        # Generate plots
        plot_results(all_results, "psnr",
                     os.path.join(args.output, f"psnr_{channel_type}.png"),
                     channel_type)
        plot_results(all_results, "ssim",
                     os.path.join(args.output, f"ssim_{channel_type}.png"),
                     channel_type)

        # Save raw results
        json_results = {}
        for name, res in all_results.items():
            json_results[name] = {str(k): v for k, v in res.items()}

        with open(os.path.join(args.output, f"results_{channel_type}.json"), "w") as f:
            json.dump(json_results, f, indent=2)

    print(f"\nBenchmark complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
