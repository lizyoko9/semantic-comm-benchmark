"""DeepSC: Deep Semantic Communication for Text.

Based on: "Deep Learning Enabled Semantic Communication Systems"
(Xie & Qin, IEEE TSP 2021).

Architecture:
- Semantic encoder: Embedding + TransformerEncoderLayers
- Channel encoder: Linear layers + power normalization
- Channel decoder: Linear layers
- Semantic decoder: TransformerDecoderLayers + Linear → vocab_size
"""

import torch
import torch.nn as nn
import math
from models.base_model import BaseSemanticModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DeepSC(BaseSemanticModel):
    """DeepSC: Transformer-based text semantic communication.

    Args:
        channel: Channel module.
        vocab_size: Vocabulary size.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers (encoder and decoder).
        channel_dim: Channel coding dimension.
        max_len: Maximum sequence length.
    """

    def __init__(self, channel, vocab_size: int = 10000, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3, channel_dim: int = 32,
                 max_len: int = 30):
        super().__init__(channel)
        self.d_model = d_model
        self.channel_dim = channel_dim
        self.max_len = max_len

        # Semantic encoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.semantic_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Channel encoder: d_model → 256 → channel_dim
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, channel_dim),
        )

        # Channel decoder: channel_dim → 256 → d_model
        self.channel_decoder = nn.Sequential(
            nn.Linear(channel_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_model),
        )

        # Semantic decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.semantic_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode token sequence to channel symbols.

        Args:
            x: [B, S] token indices.

        Returns:
            z: [B, S * channel_dim] channel symbols.
        """
        # Create padding mask
        pad_mask = (x == 0)  # True where padding

        # Semantic encoding
        emb = self.embedding(x) * math.sqrt(self.d_model)  # [B, S, d_model]
        emb = self.pos_enc(emb)
        semantic = self.semantic_encoder(emb, src_key_padding_mask=pad_mask)

        # Channel encoding (per-position)
        z = self.channel_encoder(semantic)  # [B, S, channel_dim]
        B, S, C = z.shape
        z = z.reshape(B, S * C)            # [B, S * channel_dim]
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode channel symbols to logits.

        Args:
            z: [B, S * channel_dim]

        Returns:
            logits: [B, S, vocab_size]
        """
        B = z.shape[0]
        S = self.max_len
        C = self.channel_dim
        z = z.reshape(B, S, C)

        # Channel decoding
        memory = self.channel_decoder(z)    # [B, S, d_model]

        # Semantic decoding (auto-regressive style, but using full memory)
        tgt = memory  # Use received features as decoder input
        decoded = self.semantic_decoder(tgt, memory)  # [B, S, d_model]

        logits = self.output_proj(decoded)   # [B, S, vocab_size]
        return logits

    def forward(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Full forward pass.

        Args:
            x: [B, S] token indices.
            snr_db: Channel SNR in dB.

        Returns:
            logits: [B, S, vocab_size]
        """
        z = self.encode(x)
        z = self.power_normalize(z)
        z = self.channel(z, snr_db)
        logits = self.decode(z)
        return logits
