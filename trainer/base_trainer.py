"""Abstract base trainer with train/validation loop, checkpointing, logging."""

import os
import torch
from abc import abstractmethod
from utils.logger import Logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class BaseTrainer:
    """Base trainer for semantic communication models.

    Provides:
    - Training loop with validation
    - Checkpoint save/load
    - TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(self, model, train_loader, val_loader, config: dict,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.0),
        )

        # Scheduler
        scheduler_type = config.get("scheduler", "cosine")
        epochs = config.get("epochs", 200)
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=epochs // 3, gamma=0.1
            )
        else:
            self.scheduler = None

        # Logging
        model_name = config.get("model_name", "model")
        log_dir = os.path.join(config.get("log_dir", "./runs"), model_name)
        self.logger = Logger(log_dir, name=model_name)

        # Checkpointing
        self.checkpoint_dir = os.path.join(
            config.get("checkpoint_dir", "./checkpoints"), model_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every = config.get("save_every", 20)

        self.start_epoch = 0
        self.best_metric = float('inf')

    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        epoch, metrics = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.device
        )
        self.start_epoch = epoch + 1
        self.best_metric = metrics.get("best_metric", float('inf'))
        self.logger.info(f"Resumed from epoch {epoch}")

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch. Returns dict of metrics."""
        ...

    @abstractmethod
    def validate(self, epoch: int) -> dict:
        """Validate model. Returns dict of metrics."""
        ...

    def train(self):
        """Full training loop."""
        epochs = self.config.get("epochs", 200)

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            train_metrics = self.train_epoch(epoch)

            self.model.eval()
            val_metrics = self.validate(epoch)

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.scalar("lr", lr, epoch)
            for k, v in train_metrics.items():
                self.logger.scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.logger.scalar(f"val/{k}", v, epoch)

            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {self._fmt(train_metrics)} | "
                f"Val: {self._fmt(val_metrics)} | "
                f"LR: {lr:.2e}"
            )

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpoint
            val_loss = val_metrics.get("loss", float('inf'))
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss

            if (epoch + 1) % self.save_every == 0 or is_best:
                metrics = {**val_metrics, "best_metric": self.best_metric}
                path = os.path.join(
                    self.checkpoint_dir,
                    f"best.pt" if is_best else f"epoch_{epoch}.pt"
                )
                save_checkpoint(self.model, self.optimizer, epoch, metrics, path)
                if is_best:
                    self.logger.info(f"New best model saved (loss={val_loss:.4f})")

        self.logger.info("Training complete.")
        self.logger.close()

    @staticmethod
    def _fmt(metrics: dict) -> str:
        return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
