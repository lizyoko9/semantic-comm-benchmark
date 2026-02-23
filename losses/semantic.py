"""Semantic loss for text communication (cross-entropy)."""

import torch
import torch.nn as nn


class SemanticCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for text semantic communication.

    Ignores padding tokens (index 0) by default.
    """

    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Compute cross-entropy loss.

        Args:
            logits: [B, seq_len, vocab_size]
            targets: [B, seq_len] token indices
        """
        B, S, V = logits.shape
        return self.criterion(logits.reshape(B * S, V), targets.reshape(B * S))
