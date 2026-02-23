# semantic-comm-benchmark

A PyTorch benchmark framework for comparing deep learning-based semantic communication models against traditional separate source-channel coding baselines.

## Overview

semantic-comm-benchmark reproduces the baseline comparisons commonly seen in semantic communication papers, providing a unified evaluation framework with consistent datasets, channel models, and metrics.

### Supported Models

| Model | Type | Key Feature | Paper |
|-------|------|-------------|-------|
| **DeepJSCC** | Image | CNN autoencoder, fixed-SNR | Bourtsoulatze et al., IEEE TCCN 2019 |
| **ADJSCC** | Image | SNR-adaptive attention + GDN | Xu et al., IEEE JSAC 2021 |
| **NTSCC** | Image | Hyperprior + rate-distortion | Dai et al., IEEE JSAC 2022 |
| **WITT** | Image | Swin Transformer + ModNet | Yang et al., IEEE JSAC 2023 |
| **DeepSC** | Text | Transformer semantic coding | Xie & Qin, IEEE TSP 2021 |
| **JPEG+LDPC** | Image | Traditional baseline | — |
| **BPG+LDPC** | Image | Traditional baseline | — |

### Channel Models

- **AWGN** — Additive White Gaussian Noise
- **Rayleigh** — Rayleigh fading with perfect CSI equalization
- **Rician** — Rician fading with configurable K-factor

## Project Structure

```
semantic-comm-benchmark/
├── channel/              # Channel models (AWGN, Rayleigh, Rician)
├── data/                 # Dataset loaders (CIFAR-10, Kodak24, Europarl)
├── losses/               # Loss functions (MSE, rate-distortion, LPIPS, CE)
├── metrics/              # Evaluation metrics (PSNR, SSIM, LPIPS, BLEU)
├── models/
│   ├── image/            # DeepJSCC, ADJSCC, NTSCC, WITT
│   ├── text/             # DeepSC
│   └── layers/           # Shared layers (GDN, Swin Transformer blocks)
├── trainer/              # Training infrastructure
├── baselines/            # JPEG+LDPC, BPG+LDPC
├── configs/              # YAML configuration files
├── scripts/              # Entry-point scripts
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
└── utils/                # SNR utils, logging, checkpointing
```

## Quick Start

### Installation

```bash
git clone https://github.com/lizyoko9/semantic-comm-benchmark.git
cd semantic-comm-benchmark
pip install -r requirements.txt
```

### Train a Model

```bash
# Train DeepJSCC on CIFAR-10
python scripts/train.py --config configs/deepjscc.yaml

# Train ADJSCC (SNR-adaptive)
python scripts/train.py --config configs/adjscc.yaml

# Train WITT (Swin Transformer)
python scripts/train.py --config configs/witt.yaml --device cuda:0

# Resume training from checkpoint
python scripts/train.py --config configs/deepjscc.yaml --resume checkpoints/deepjscc/best.pt
```

### Evaluate a Trained Model

```bash
# Evaluate at multiple SNR points
python scripts/evaluate.py \
    --config configs/deepjscc.yaml \
    --checkpoint checkpoints/deepjscc/best.pt \
    --snr -2 0 4 8 12 16 20

# Save results to JSON
python scripts/evaluate.py \
    --config configs/adjscc.yaml \
    --checkpoint checkpoints/adjscc/best.pt \
    --output results/adjscc_awgn.json
```

### Run Full Benchmark

```bash
# Compare all trained models + traditional baselines
python scripts/benchmark.py \
    --models deepjscc adjscc ntscc witt \
    --channels awgn \
    --snr -2 0 2 4 6 8 10 12 14 16 18 20 \
    --output results/
```

This generates:
- PSNR vs SNR and SSIM vs SNR plots
- Result tables printed to console
- Raw results in JSON format

## Configuration

Each model has a YAML config file in `configs/`. Key parameters:

```yaml
model_name: deepjscc
model:
  name: deepjscc
  c_out: 16              # Controls bandwidth ratio

channel_type: awgn       # awgn | rayleigh | rician
snr_train: 10.0          # Training SNR (dB)
random_snr: false        # Sample random SNR during training

batch_size: 128
lr: 1.0e-3
epochs: 200
```

## Model Details

| Model | Parameters | Bandwidth Ratio | SNR-Adaptive |
|-------|-----------|----------------|--------------|
| DeepJSCC | 156K | ~1/3 (c_out=16) | No |
| ADJSCC | 167K | ~1/3 (c_out=16) | Yes |
| NTSCC | 2.5M | Configurable (k) | No |
| WITT | 12.7M | Configurable | Yes |
| DeepSC | 4.0M | Configurable | No |

## Metrics

- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index
- **LPIPS** — Learned Perceptual Image Patch Similarity
- **BLEU** — Bilingual Evaluation Understudy (for text)

## Citation

If you use this benchmark in your research, please cite the relevant original papers for each model.

## License

MIT
