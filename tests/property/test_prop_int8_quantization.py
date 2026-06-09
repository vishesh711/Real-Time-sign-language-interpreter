"""
Property-based test for INT8 quantization agreement with FP32.

**Feature: sign-language-interpreter, Property 13: INT8 quantization agreement with FP32**
**Validates: Requirements 7.2**

For any valid landmark input from the test distribution, the INT8-quantized ONNX
model's top-1 predicted class SHALL agree with the FP32 model's top-1 predicted
class at a rate >= 98% over a sample of >= 100 inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from onnxruntime.quantization import QuantType, quantize_dynamic

from models.mlp import FingerspellingMLP

# ---------------------------------------------------------------------------
# Module-level fixture: build FP32 and INT8 sessions once
# ---------------------------------------------------------------------------

_INPUT_DIM = 63
_NUM_CLASSES = 36


def _export_fp32_legacy(model: torch.nn.Module, output_path: Path, input_dim: int) -> None:
    """Export using the legacy TorchScript-based exporter (dynamo=False).

    The new dynamo exporter in PyTorch 2.9+ produces graphs whose batch-axis
    symbolic dim confuses ORT's shape-inference step inside quantize_dynamic.
    The legacy exporter avoids that issue and is fully supported for MLP models.
    """
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    proto = onnx.load(str(output_path))
    onnx.checker.check_model(proto)


def _setup_sessions() -> tuple[ort.InferenceSession, ort.InferenceSession]:
    """Export FP32 ONNX model, quantize to INT8, and load both as ORT sessions."""
    torch.manual_seed(0)
    model = FingerspellingMLP(
        input_dim=_INPUT_DIM,
        hidden_dims=(256, 256, 128),
        num_classes=_NUM_CLASSES,
        dropout=0.0,  # deterministic at eval time
    )
    model.eval()

    tmp_dir = Path(tempfile.mkdtemp(prefix="int8_quant_test_"))
    fp32_path = tmp_dir / "fingerspell_fp32.onnx"
    int8_path = tmp_dir / "fingerspell_int8.onnx"

    _export_fp32_legacy(model, fp32_path, _INPUT_DIM)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    fp32_session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    return fp32_session, int8_session


_FP32_SESSION, _INT8_SESSION = _setup_sessions()

# ---------------------------------------------------------------------------
# Accumulators for overall agreement rate across all Hypothesis examples
# ---------------------------------------------------------------------------

_agreement_count = 0
_total_count = 0

# ---------------------------------------------------------------------------
# Strategy: valid normalized landmark inputs
# ---------------------------------------------------------------------------


@st.composite
def landmark_samples(draw) -> np.ndarray:
    """
    Generate a batch of 1–4 landmark vectors with values in [-3.0, 3.0].
    Batches are kept small to stay within Hypothesis entropy budget while
    still accumulating >= 100 individual sample results across 100 runs.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    flat = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=batch_size * _INPUT_DIM,
            max_size=batch_size * _INPUT_DIM,
        )
    )
    return np.array(flat, dtype=np.float32).reshape(batch_size, _INPUT_DIM)


# ---------------------------------------------------------------------------
# Property 13: INT8 quantization agreement with FP32
# **Feature: sign-language-interpreter, Property 13: INT8 quantization agreement with FP32**
# **Validates: Requirements 7.2**
# ---------------------------------------------------------------------------


@given(x=landmark_samples())
@settings(max_examples=100)
def test_int8_top1_agrees_with_fp32(x: np.ndarray) -> None:
    """
    **Feature: sign-language-interpreter, Property 13: INT8 quantization agreement with FP32**
    **Validates: Requirements 7.2**

    For any valid landmark input, the INT8-quantized ONNX model's top-1 predicted
    class SHALL agree with the FP32 model's top-1 predicted class at a rate >= 98%
    over the full sample of >= 100 inputs drawn by Hypothesis.
    """
    global _agreement_count, _total_count

    fp32_input_name = _FP32_SESSION.get_inputs()[0].name
    int8_input_name = _INT8_SESSION.get_inputs()[0].name

    fp32_logits = _FP32_SESSION.run(None, {fp32_input_name: x})[0]  # (B, 36)
    int8_logits = _INT8_SESSION.run(None, {int8_input_name: x})[0]  # (B, 36)

    fp32_top1 = np.argmax(fp32_logits, axis=1)
    int8_top1 = np.argmax(int8_logits, axis=1)

    batch_size = x.shape[0]
    batch_agreements = int(np.sum(fp32_top1 == int8_top1))

    _agreement_count += batch_agreements
    _total_count += batch_size

    # Enforce the >=98% agreement rate using the running totals so far
    overall_rate = _agreement_count / _total_count
    assert overall_rate >= 0.98, (
        f"INT8 top-1 agreement rate {overall_rate:.4f} fell below 98% threshold "
        f"({_agreement_count}/{_total_count} agreed). "
        f"Current batch — FP32 top-1: {fp32_top1.tolist()}, "
        f"INT8 top-1: {int8_top1.tolist()}"
    )
