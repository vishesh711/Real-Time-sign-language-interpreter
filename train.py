"""
Fingerspelling training script.

Usage:
    python train.py --csv data/sign_mnist_train.csv --val-csv data/sign_mnist_test.csv
    python train.py --csv data/sign_mnist_train.csv  # auto stratified 80/20 split

Features:
  - Stratified 80/20 train/val split (or explicit val CSV)
  - Cosine LR with warm restarts (CosineAnnealingWarmRestarts)
  - Label smoothing (0.1)
  - Early stopping on val accuracy
  - Per-class accuracy breakdown at the end

Requirements: 2.1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from models.mlp import FingerspellingMLP, FingerspellingMLPSmall
from utils.dataset import LandmarkDataset, load_sign_mnist_csv
from utils.landmarks import IDX_TO_LABEL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FingerspellingMLP")
    p.add_argument("--csv", required=True, help="Path to Sign Language MNIST train CSV")
    p.add_argument("--val-csv", default=None, help="Optional explicit validation CSV")
    p.add_argument("--model", choices=["full", "small"], default="full")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _make_split(dataset: LandmarkDataset, val_frac: float = 0.2, seed: int = 42):
    """Stratified 80/20 split returning (train_subset, val_subset)."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    indices = np.arange(len(dataset))
    train_idx, val_idx = next(splitter.split(indices, dataset.labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[int, float]]:
    """Return overall accuracy and per-class accuracy dict."""
    model.eval()
    correct_per_class: dict[int, int] = {}
    total_per_class: dict[int, int] = {}
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for pred, gt in zip(preds.cpu().numpy(), y.cpu().numpy()):
                total_per_class[gt] = total_per_class.get(gt, 0) + 1
                if pred == gt:
                    correct_per_class[gt] = correct_per_class.get(gt, 0) + 1
                    total_correct += 1
                total_samples += 1

    overall = total_correct / max(total_samples, 1)
    per_class = {
        cls: correct_per_class.get(cls, 0) / cnt
        for cls, cnt in total_per_class.items()
    }
    return overall, per_class


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    train_full = load_sign_mnist_csv(args.csv, augment=True)
    num_classes = train_full.num_classes

    if args.val_csv:
        val_ds = load_sign_mnist_csv(args.val_csv, augment=False)
        train_ds = train_full
    else:
        train_ds, val_ds = _make_split(train_full, val_frac=0.2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {num_classes}")

    # Model
    if args.model == "small":
        model = FingerspellingMLPSmall(input_dim=63, num_classes=num_classes, dropout=args.dropout)
    else:
        model = FingerspellingMLP(input_dim=63, num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser + cosine LR with warm restarts
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(x)

        scheduler.step()
        avg_loss = running_loss / max(len(train_ds), 1)

        val_acc, _ = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = output_dir / "best_fingerspell.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "label_to_idx": {v: k for k, v in IDX_TO_LABEL.items()},
                    "idx_to_label": IDX_TO_LABEL,
                    "hidden_dims": list(model.blocks[0][0].out_features if hasattr(model.blocks[0][0], "out_features") else [256, 256, 128]),
                    "num_classes": num_classes,
                    "dropout": args.dropout,
                    "model_type": args.model,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Per-class accuracy breakdown
    best_ckpt = output_dir / "best_fingerspell.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])

    val_acc_final, per_class_acc = _evaluate(model, val_loader, device)
    print("\nPer-class accuracy breakdown:")
    for cls_idx in sorted(per_class_acc.keys()):
        label = IDX_TO_LABEL.get(cls_idx, str(cls_idx))
        print(f"  {label:>3s}: {per_class_acc[cls_idx]:.4f}")


if __name__ == "__main__":
    train()
