"""
FingerspellMLP — static-pose fingerspelling classifier.

Architecture (per block): Linear → BatchNorm1d → ReLU → Dropout
Three blocks followed by a final Linear output layer.

Requirements: 2.1, 2.2
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FingerspellingMLP(nn.Module):
    """Configurable MLP for ASL fingerspelling classification.

    Args:
        input_dim:   Number of input features (default 63 = 21 landmarks × 3).
        hidden_dims: Sequence of three hidden layer widths.
        num_classes: Number of output classes (default 36 = A–Z + 0–9).
        dropout:     Dropout probability applied after each hidden block.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dims: Sequence[int] = (256, 256, 128),
        num_classes: int = 36,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if len(hidden_dims) != 3:
            raise ValueError(f"hidden_dims must have exactly 3 elements, got {len(hidden_dims)}")

        def _block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )

        self.blocks = nn.Sequential(
            _block(input_dim, hidden_dims[0]),
            _block(hidden_dims[0], hidden_dims[1]),
            _block(hidden_dims[1], hidden_dims[2]),
        )
        self.head = nn.Linear(hidden_dims[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, input_dim).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.head(self.blocks(x))


class FingerspellingMLPSmall(FingerspellingMLP):
    """Lightweight variant optimised for INT8 ONNX deployment in the browser.

    Smaller hidden dimensions reduce model size while maintaining accuracy on
    the 26-class fingerspelling task.
    """

    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 36,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=(128, 128, 64),
            num_classes=num_classes,
            dropout=dropout,
        )
