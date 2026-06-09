"""
Word-level sign language training script (SignLSTM).

Usage:
    python train_word.py --wlasl-root data/wlasl_landmarks --asl-root data/asl_citizen_landmarks --asl-csv data/asl_citizen_metadata.csv
    python train_word.py --wlasl-root data/wlasl_landmarks  # WLASL only, no ASL Citizen fine-tuning

Features:
  - Pre-training on WLASL-300 with WeightedRandomSampler for class imbalance
  - Fine-tuning on ASL Citizen with pre-defined signer-independent splits
  - OneCycleLR scheduler
  - Mixed precision (--amp flag)
  - Top-1 / Top-5 accuracy evaluation on ASL Citizen signer-independent test split
  - Per-class accuracy breakdown

Requirements: 10.1, 10.2, 10.4, 10.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.cnn3d_lstm import SignLSTM
from utils.sequence_dataset import (
    SignSequenceDataset,
    load_asl_citizen_dataset,
    load_wlasl_dataset,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SignLSTM for word-level ASL recognition")
    p.add_argument("--wlasl-root", default=None,
                   help="Root of pre-extracted WLASL landmark .npy files")
    p.add_argument("--asl-root", default=None,
                   help="Root of pre-extracted ASL Citizen landmark .npy files")
    p.add_argument("--asl-csv", default=None,
                   help="Path to ASL Citizen metadata CSV (required with --asl-root)")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision training (FP16)")
    p.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# WeightedRandomSampler builder
# ---------------------------------------------------------------------------

def _make_weighted_sampler(dataset: SignSequenceDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that compensates for class imbalance.

    Each sample's weight is the inverse frequency of its class so that every
    class is sampled at approximately equal rate per epoch.

    Requirements: 10.4
    """
    labels = dataset.labels  # (N,) int array
    num_classes = dataset.num_classes
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    # Avoid division by zero for absent classes
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k: int = 5,
) -> tuple[float, float, Dict[int, float]]:
    """Compute top-1, top-5 accuracy and per-class accuracy.

    Requirements: 10.2, 10.5
    """
    model.eval()
    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0
    correct_per_class: Dict[int, int] = {}
    total_per_class: Dict[int, int] = {}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, num_classes)

            # Top-1
            top1_preds = logits.argmax(dim=1)
            total_correct_top1 += int((top1_preds == y).sum())

            # Top-k
            k = min(top_k, logits.size(1))
            topk_preds = logits.topk(k, dim=1).indices  # (B, k)
            for i, gt in enumerate(y.cpu().numpy()):
                if gt in topk_preds[i].cpu().numpy():
                    total_correct_topk += 1

            # Per-class
            for pred, gt in zip(top1_preds.cpu().numpy(), y.cpu().numpy()):
                total_per_class[int(gt)] = total_per_class.get(int(gt), 0) + 1
                if pred == gt:
                    correct_per_class[int(gt)] = correct_per_class.get(int(gt), 0) + 1

            total_samples += len(y)

    top1 = total_correct_top1 / max(total_samples, 1)
    topk_acc = total_correct_topk / max(total_samples, 1)
    per_class_acc = {
        cls: correct_per_class.get(cls, 0) / cnt
        for cls, cnt in total_per_class.items()
    }
    return top1, topk_acc, per_class_acc


def _print_per_class_breakdown(
    per_class_acc: Dict[int, float],
    idx_to_label: Dict[int, str],
    top_n: int = 20,
) -> None:
    """Print per-class accuracy, showing the N worst-performing classes first.

    Requirements: 10.5
    """
    sorted_classes = sorted(per_class_acc.items(), key=lambda kv: kv[1])
    print("\nPer-class accuracy breakdown (worst first):")
    for cls_idx, acc in sorted_classes[:top_n]:
        label = idx_to_label.get(cls_idx, str(cls_idx))
        print(f"  {label:>20s}: {acc:.4f}")
    if len(sorted_classes) > top_n:
        print(f"  ... ({len(sorted_classes) - top_n} more classes not shown)")


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional["torch.cuda.amp.GradScaler"],  # noqa: F821
) -> float:
    """Run one training epoch, returning the mean loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * len(x)
        total_samples += len(x)

    return running_loss / max(total_samples, 1)


def _run_training(
    model: nn.Module,
    train_ds: SignSequenceDataset,
    val_ds: SignSequenceDataset,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_name: str,
    use_weighted_sampler: bool = False,
) -> None:
    """Generic training loop used for both pre-training and fine-tuning phases."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_weighted_sampler:
        sampler = _make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples | "
          f"Classes: {train_ds.num_classes}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    # Mixed precision scaler (CUDA only)
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed-precision (FP16) enabled.")

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()

        val_top1, val_top5, _ = _evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_top1={val_top1:.4f}  val_top5={val_top5:.4f}  lr={lr_now:.2e}")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            patience_counter = 0
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "label_to_idx": train_ds.label_to_idx,
                "idx_to_label": train_ds.idx_to_label,
                "seq_len": args.seq_len,
                "feature_dim": 252,
                "use_velocity": True,
                "num_classes": train_ds.num_classes,
            }
            torch.save(ckpt, output_dir / checkpoint_name)
            print(f"  ✓ Saved checkpoint (val_top1={val_top1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"Best val top-1: {best_val_acc:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(argv=None) -> None:
    args = _parse_args(argv)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.wlasl_root is None and args.asl_root is None:
        raise ValueError(
            "At least one of --wlasl-root or --asl-root must be provided."
        )

    # -----------------------------------------------------------------------
    # Phase 1: Pre-training on WLASL-300  (Requirements: 10.1, 10.4)
    # -----------------------------------------------------------------------
    if args.wlasl_root:
        print("\n=== Phase 1: Pre-training on WLASL-300 ===")
        wlasl_train = load_wlasl_dataset(args.wlasl_root, split="train",
                                         seq_len=args.seq_len, augment=True)
        wlasl_val_dir = Path(args.wlasl_root) / "val"
        if wlasl_val_dir.exists():
            wlasl_val = load_wlasl_dataset(args.wlasl_root, split="val",
                                           seq_len=args.seq_len, augment=False)
        else:
            # Fall back: use 10% of train as val
            from torch.utils.data import random_split
            n_val = max(1, len(wlasl_train) // 10)
            wlasl_train, wlasl_val = random_split(
                wlasl_train, [len(wlasl_train) - n_val, n_val]
            )
            # Wrap in a minimal proxy so .label_to_idx is accessible
            wlasl_val.label_to_idx = wlasl_train.dataset.label_to_idx
            wlasl_val.idx_to_label = wlasl_train.dataset.idx_to_label
            wlasl_val.num_classes = wlasl_train.dataset.num_classes

        num_classes = wlasl_train.num_classes

        model = SignLSTM(
            seq_len=args.seq_len,
            feature_dim=252,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)

        _run_training(
            model=model,
            train_ds=wlasl_train,
            val_ds=wlasl_val,
            args=args,
            device=device,
            checkpoint_name="best_word_wlasl.pt",
            use_weighted_sampler=True,  # compensate class imbalance (Req 10.4)
        )

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tuning on ASL Citizen signer-independent split
    #           (Requirements: 10.1, 10.2, 10.3)
    # -----------------------------------------------------------------------
    if args.asl_root and args.asl_csv:
        print("\n=== Phase 2: Fine-tuning on ASL Citizen ===")

        asl_train = load_asl_citizen_dataset(
            landmark_root=args.asl_root,
            metadata_csv=args.asl_csv,
            split="train",
            seq_len=args.seq_len,
            augment=True,
        )
        asl_val = load_asl_citizen_dataset(
            landmark_root=args.asl_root,
            metadata_csv=args.asl_csv,
            split="val",
            seq_len=args.seq_len,
            augment=False,
        )
        asl_test = load_asl_citizen_dataset(
            landmark_root=args.asl_root,
            metadata_csv=args.asl_csv,
            split="test",
            seq_len=args.seq_len,
            augment=False,
        )

        num_classes = asl_train.num_classes

        # Load WLASL checkpoint if available, or start fresh
        wlasl_ckpt = Path(args.output_dir) / "best_word_wlasl.pt"
        if wlasl_ckpt.exists():
            print(f"Loading WLASL pre-trained weights from {wlasl_ckpt}")
            ckpt = torch.load(wlasl_ckpt, map_location=device)
            wlasl_classes = len(ckpt["label_to_idx"])
            # Build model with WLASL head first, then replace head for ASL Citizen
            model = SignLSTM(
                seq_len=args.seq_len, feature_dim=252,
                num_classes=wlasl_classes, dropout=args.dropout,
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            # Replace classifier head for the new class count
            model.head = nn.Linear(model.head.in_features, num_classes).to(device)
        else:
            model = SignLSTM(
                seq_len=args.seq_len, feature_dim=252,
                num_classes=num_classes, dropout=args.dropout,
            ).to(device)

        _run_training(
            model=model,
            train_ds=asl_train,
            val_ds=asl_val,
            args=args,
            device=device,
            checkpoint_name="best_word_asl_citizen.pt",
            use_weighted_sampler=False,
        )

        # -----------------------------------------------------------------------
        # Final evaluation on signer-independent test split (Requirement: 10.2)
        # -----------------------------------------------------------------------
        best_ckpt = Path(args.output_dir) / "best_word_asl_citizen.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state["model_state"])

        test_loader = DataLoader(asl_test, batch_size=64, shuffle=False, num_workers=0)
        test_top1, test_top5, per_class_acc = _evaluate(model, test_loader, device)

        print(f"\n=== ASL Citizen Signer-Independent Test Set Results ===")
        print(f"  Top-1 accuracy: {test_top1:.4f}")
        print(f"  Top-5 accuracy: {test_top5:.4f}")

        _print_per_class_breakdown(per_class_acc, asl_test.idx_to_label)

    elif args.asl_root and not args.asl_csv:
        print("Warning: --asl-root provided without --asl-csv; skipping ASL Citizen phase.")


if __name__ == "__main__":
    train()
