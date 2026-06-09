"""
SignLSTM — word-level sign language classifier.

Architecture (per design doc):
  Input [B, T=30, F=252]
    → Reshape each frame to [B, T, 1, 16, 16]  (zero-pad F to 256)
    → Conv3DBlock × 3  (channels: 1→32→64→128, temporal: 30→15)
    → Reshape to [B, 15, 128*H*W]
    → BiLSTM(hidden=256, layers=2, dropout=0.3)
    → TemporalAttention  (weighted sum over 15 timesteps)
    → LayerNorm → Dropout(0.4) → Linear(512→300)

Requirements: 3.1, 3.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """3D convolution block: Conv3d → BatchNorm3d → ReLU → MaxPool3d (temporal dim only).

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        reduce_time:  If True, apply a max-pool that halves the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduce_time: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Pool only the temporal dimension (depth); keep spatial dims stable
        if reduce_time:
            self.pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, T, H, W).
        Returns:
            Tensor of shape (B, out_channels, T', H, W).
        """
        return self.pool(self.relu(self.bn(self.conv(x))))


class TemporalAttention(nn.Module):
    """Scalar attention over T time steps of LSTM output.

    Computes a softmax weight for each time step and returns the
    weighted sum, collapsing the temporal dimension.

    Args:
        hidden_dim: Dimension of each LSTM output vector (BiLSTM → 2 × hidden).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Tensor of shape (B, T, hidden_dim).
        Returns:
            Context vector of shape (B, hidden_dim).
        """
        # (B, T, 1) → (B, T) attention weights
        weights = F.softmax(self.score(h).squeeze(-1), dim=1)  # (B, T)
        # Weighted sum: (B, T) × (B, T, D) → (B, D)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context


class SignLSTM(nn.Module):
    """Word-level sign language classifier using 3D CNN + BiLSTM + Temporal Attention.

    Args:
        seq_len:       Number of input frames T (default 30).
        feature_dim:   Input feature dimension F (default 252).
        num_classes:   Number of output gloss classes (default 300 for WLASL-300).
        cnn_channels:  Channel progression for the three Conv3DBlock layers.
        lstm_hidden:   BiLSTM hidden units per direction (combined = 2 × this).
        lstm_layers:   Number of stacked BiLSTM layers.
        lstm_dropout:  Dropout between LSTM layers.
        dropout:       Dropout before the final classifier head.
        spatial_size:  Spatial H=W after reshaping the feature vector (default 16).
    """

    def __init__(
        self,
        seq_len: int = 30,
        feature_dim: int = 252,
        num_classes: int = 300,
        cnn_channels: tuple[int, int, int] = (32, 64, 128),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        dropout: float = 0.4,
        spatial_size: int = 16,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self._padded_dim = spatial_size * spatial_size  # 256

        # 3-block CNN: channels 1 → 32 → 64 → 128
        # First block halves temporal; blocks 2-3 maintain
        self.cnn = nn.Sequential(
            Conv3DBlock(1, cnn_channels[0], reduce_time=True),   # T → T//2
            Conv3DBlock(cnn_channels[0], cnn_channels[1], reduce_time=False),
            Conv3DBlock(cnn_channels[1], cnn_channels[2], reduce_time=False),
        )

        # After CNN: [B, 128, T//2, H, W]
        cnn_out_time = seq_len // 2
        cnn_feat_dim = cnn_channels[2] * spatial_size * spatial_size

        # BiLSTM: input = cnn_feat_dim, output = 2 * lstm_hidden per step
        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Temporal attention collapses (B, T', lstm_out_dim) → (B, lstm_out_dim)
        self.attention = TemporalAttention(lstm_out_dim)

        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, F) — T frames, F features per frame.
        Returns:
            Logits tensor of shape (B, num_classes).
        """
        B, T, F = x.shape

        # Zero-pad feature dimension from F to spatial_size²
        pad_size = self._padded_dim - F
        if pad_size > 0:
            padding = torch.zeros(B, T, pad_size, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, padding], dim=-1)  # (B, T, 256)
        elif pad_size == 0:
            x_pad = x
        else:
            x_pad = x[:, :, :self._padded_dim]

        # Reshape to 3D video: (B, 1, T, H, W)
        x_3d = x_pad.reshape(B, T, 1, self.spatial_size, self.spatial_size)
        x_3d = x_3d.permute(0, 2, 1, 3, 4)  # (B, C=1, T, H, W)

        # 3D CNN
        cnn_out = self.cnn(x_3d)  # (B, 128, T', H, W)
        _, C, T2, H, W = cnn_out.shape

        # Flatten spatial and channel dims → sequence for LSTM
        lstm_in = cnn_out.permute(0, 2, 1, 3, 4).contiguous()  # (B, T', C, H, W)
        lstm_in = lstm_in.view(B, T2, C * H * W)  # (B, T', feat)

        # BiLSTM
        lstm_out, _ = self.lstm(lstm_in)  # (B, T', 2*hidden)

        # Temporal attention
        context = self.attention(lstm_out)  # (B, 2*hidden)

        # Head
        out = self.dropout(self.norm(context))
        return self.head(out)  # (B, num_classes)
