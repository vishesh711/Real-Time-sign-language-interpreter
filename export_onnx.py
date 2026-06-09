"""
ONNX export and INT8 quantization pipeline for the Sign Language Interpreter models.

Exports a trained PyTorch checkpoint to:
  1. A full-precision ONNX model (opset 17, dynamic batch axis).
  2. An INT8 dynamically-quantized ONNX model via onnxruntime.quantization.
  3. A JSON labels sidecar file mapping index → class label.

Usage:
    python export_onnx.py --checkpoint checkpoints/best_fingerspell.pt \
                          --output-dir exports/

    python export_onnx.py --checkpoint checkpoints/best_fingerspell.pt \
                          --output-dir exports/ \
                          --model-type small

Requirements: 7.1, 7.2, 7.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

from models.mlp import FingerspellingMLP, FingerspellingMLPSmall
from utils.landmarks import IDX_TO_LABEL, LABEL_TO_IDX


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export FingerspellingMLP to ONNX + INT8")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint saved by train.py",
    )
    p.add_argument(
        "--output-dir",
        default="exports",
        help="Directory to write the ONNX model and labels sidecar (default: exports/)",
    )
    p.add_argument(
        "--model-type",
        choices=["full", "small"],
        default=None,
        help="Model variant; if omitted, inferred from the checkpoint",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    p.add_argument(
        "--input-dim",
        type=int,
        default=63,
        help="Number of input features (default: 63 = 21 landmarks × 3)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization step",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(
    checkpoint_path: Path,
    model_type: str | None,
    input_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load a FingerspellingMLP from a checkpoint file.

    Returns:
        (model, checkpoint_dict) — model is in eval mode on *device*.
    """
    state = torch.load(str(checkpoint_path), map_location=device)

    # Infer model type and num_classes from checkpoint when not provided
    resolved_type = model_type or state.get("model_type", "full")
    num_classes = state.get("num_classes", len(IDX_TO_LABEL))
    dropout = state.get("dropout", 0.3)

    if resolved_type == "small":
        model = FingerspellingMLPSmall(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        hidden_dims = state.get("hidden_dims", [256, 256, 128])
        # Ensure we have exactly 3 hidden dims
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) != 3:
            hidden_dims = [256, 256, 128]
        model = FingerspellingMLP(
            input_dim=input_dim,
            hidden_dims=tuple(hidden_dims),
            num_classes=num_classes,
            dropout=dropout,
        )

    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    return model, state


# ---------------------------------------------------------------------------
# ONNX export  (Requirement 7.1)
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_dim: int,
    opset: int = 17,
) -> None:
    """Export a PyTorch model to ONNX with a dynamic batch axis.

    The exported model has:
      - Input:  'landmarks'  — shape (batch, input_dim)
      - Output: 'logits'     — shape (batch, num_classes)

    Runs onnx.checker.check_model to validate the graph.

    Args:
        model:       Model in eval mode.
        output_path: Destination .onnx file path.
        input_dim:   Number of input features.
        opset:       ONNX opset version (default 17).

    Raises:
        onnx.checker.ValidationError: If the exported model is invalid.
    """
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # Validate with ONNX model checker (Requirement 7.1)
    proto = onnx.load(str(output_path))
    onnx.checker.check_model(proto)
    print(f"  ✓  ONNX model written and validated: {output_path}")


# ---------------------------------------------------------------------------
# INT8 dynamic quantization  (Requirement 7.2)
# ---------------------------------------------------------------------------

def quantize_to_int8(fp32_onnx_path: Path, int8_onnx_path: Path) -> None:
    """Apply INT8 dynamic quantization to a full-precision ONNX model.

    Uses onnxruntime.quantization.quantize_dynamic which quantizes MatMul
    and GEMM operators to INT8 without requiring a calibration dataset.
    This is suitable for the linear-only FingerspellingMLP architecture.

    Args:
        fp32_onnx_path: Path to the full-precision ONNX model.
        int8_onnx_path: Destination path for the quantized model.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QInt8,
    )
    print(f"  ✓  INT8-quantized model written: {int8_onnx_path}")


# ---------------------------------------------------------------------------
# Labels sidecar  (Requirement 7.3 / design ModelCheckpoint spec)
# ---------------------------------------------------------------------------

def write_labels_sidecar(
    idx_to_label: dict[int, str],
    output_path: Path,
) -> None:
    """Write a JSON sidecar mapping index → label alongside the ONNX model.

    The sidecar is a JSON array ordered by index so ``labels[i]`` gives the
    human-readable class name for output index *i*.

    Args:
        idx_to_label: Dict mapping integer index → string label.
        output_path:  Destination .json file path.
    """
    max_idx = max(idx_to_label.keys())
    labels_list = [idx_to_label[i] for i in range(max_idx + 1)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels_list, f, ensure_ascii=False, indent=2)
    print(f"  ✓  Labels sidecar written ({len(labels_list)} classes): {output_path}")


# ---------------------------------------------------------------------------
# Serialization round-trip verification  (Requirement 7.3)
# ---------------------------------------------------------------------------

def verify_round_trip(
    model: torch.nn.Module,
    onnx_path: Path,
    input_dim: int,
    num_samples: int = 20,
    atol: float = 1e-4,
) -> None:
    """Verify that the saved ONNX model reproduces PyTorch outputs.

    Loads the ONNX file into a fresh ORT session and compares outputs to
    the original PyTorch model on random inputs.  Raises if any sample
    exceeds *atol*.

    Args:
        model:       Original PyTorch model in eval mode.
        onnx_path:   Path to the exported ONNX model.
        input_dim:   Feature dimension.
        num_samples: Number of random inputs to test.
        atol:        Maximum allowed absolute difference.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(seed=0)
    max_diff = 0.0

    for _ in range(num_samples):
        x_np = rng.standard_normal((4, input_dim)).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out = model(x_torch).numpy()

        ort_out = session.run(None, {input_name: x_np})[0]
        diff = float(np.max(np.abs(pt_out - ort_out)))
        max_diff = max(max_diff, diff)

    if max_diff > atol:
        raise RuntimeError(
            f"Round-trip verification failed: max absolute diff {max_diff:.2e} > atol {atol:.2e}"
        )
    print(f"  ✓  Round-trip verified (max abs diff = {max_diff:.2e}, atol = {atol:.2e})")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def export(argv: list[str] | None = None) -> None:
    """Full export pipeline: FP32 ONNX → validation → INT8 quantization → labels."""
    args = _parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # export always on CPU for portability

    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, ckpt_state = _load_model(checkpoint_path, args.model_type, args.input_dim, device)

    # Derive output filenames from the checkpoint stem
    stem = checkpoint_path.stem
    fp32_path = output_dir / f"{stem}.onnx"
    int8_path = output_dir / f"{stem}_int8.onnx"
    labels_path = output_dir / f"{stem}.labels.json"

    # Retrieve label map: prefer checkpoint-stored mapping, fall back to global
    idx_to_label: dict[int, str] = ckpt_state.get("idx_to_label", IDX_TO_LABEL)
    # Normalise keys to int (torch.save may restore them as strings)
    idx_to_label = {int(k): str(v) for k, v in idx_to_label.items()}

    print(f"\n[1/4] Exporting FP32 ONNX (opset {args.opset})…")
    export_to_onnx(model, fp32_path, args.input_dim, opset=args.opset)

    print("\n[2/4] Verifying serialization round-trip…")
    verify_round_trip(model, fp32_path, args.input_dim)

    if not args.no_quantize:
        print("\n[3/4] Quantizing to INT8…")
        quantize_to_int8(fp32_path, int8_path)
    else:
        print("\n[3/4] Skipping INT8 quantization (--no-quantize).")

    print("\n[4/4] Writing labels sidecar…")
    write_labels_sidecar(idx_to_label, labels_path)

    print(f"\nExport complete. Artifacts written to: {output_dir}/")
    print(f"  FP32 ONNX : {fp32_path.name}")
    if not args.no_quantize:
        print(f"  INT8 ONNX : {int8_path.name}")
    print(f"  Labels    : {labels_path.name}")


if __name__ == "__main__":
    export()
