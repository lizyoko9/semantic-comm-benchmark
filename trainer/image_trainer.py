"""Image trainer: concrete trainer for image semantic communication models."""

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from trainer.base_trainer import BaseTrainer
from metrics.psnr_ssim import compute_psnr, compute_ssim


class ImageTrainer(BaseTrainer):
    """Trainer for image JSCC models (DeepJSCC, ADJSCC, NTSCC, WITT)."""

    def __init__(self, model, train_loader, val_loader, config: dict,
                 device: str = "cuda"):
        super().__init__(model, train_loader, val_loader, config, device)
        self.snr_train = config.get("snr_train", 10.0)
        self.snr_range = config.get("snr_range", [-2, 0, 4, 8, 12, 16, 20])
        self.random_snr = config.get("random_snr", False)
        self.model_name = config.get("model_name", "model")

        # Rate-distortion lambda for NTSCC
        self.lmbda = config.get("lmbda", 0.01)
        self.use_rate_loss = config.get("use_rate_loss", False)

    def _sample_snr(self) -> float:
        if self.random_snr:
            return random.uniform(self.snr_range[0], self.snr_range[-1])
        return self.snr_train

    def train_epoch(self, epoch: int) -> dict:
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # CIFAR-10 returns (images, labels)
            else:
                x = batch
            x = x.to(self.device)
            snr_db = self._sample_snr()

            self.optimizer.zero_grad()

            if self.use_rate_loss:
                x_hat, rate = self.model(x, snr_db)
                mse_loss = F.mse_loss(x_hat, x)
                loss = mse_loss + self.lmbda * rate.mean()
            else:
                x_hat = self.model(x, snr_db)
                loss = F.mse_loss(x_hat, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", snr=f"{snr_db:.1f}")

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0

        snr_db = self.snr_train

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(self.device)

            if self.use_rate_loss:
                x_hat, rate = self.model(x, snr_db)
                loss = F.mse_loss(x_hat, x) + self.lmbda * rate.mean()
            else:
                x_hat = self.model(x, snr_db)
                loss = F.mse_loss(x_hat, x)

            x_hat = x_hat.clamp(0, 1)
            total_loss += loss.item()
            total_psnr += compute_psnr(x, x_hat)
            total_ssim += compute_ssim(x, x_hat)
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "psnr": total_psnr / n,
            "ssim": total_ssim / n,
        }

    @torch.no_grad()
    def evaluate_snr_sweep(self, snr_list: list = None):
        """Evaluate model at multiple SNR points.

        Returns dict mapping SNR → {psnr, ssim}.
        """
        if snr_list is None:
            snr_list = self.snr_range

        self.model.eval()
        results = {}

        for snr_db in snr_list:
            total_psnr = 0.0
            total_ssim = 0.0
            n_batches = 0

            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)

                if self.use_rate_loss:
                    x_hat, _ = self.model(x, snr_db)
                else:
                    x_hat = self.model(x, snr_db)

                x_hat = x_hat.clamp(0, 1)
                total_psnr += compute_psnr(x, x_hat)
                total_ssim += compute_ssim(x, x_hat)
                n_batches += 1

            n = max(n_batches, 1)
            results[snr_db] = {
                "psnr": total_psnr / n,
                "ssim": total_ssim / n,
            }

        return results
