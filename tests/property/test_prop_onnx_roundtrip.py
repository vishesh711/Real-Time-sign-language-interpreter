"""
Property-based test for ONNX model serialization round-trip.

**Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip**
**Validates: Requirements 7.3**

For any valid landmark input x, loading a saved ONNX model file and running
inference SHALL produce outputs identical (within floating-point epsilon of 1e-5)
to the outputs produced by the original ONNX session before saving.
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

from export_onnx import export_to_onnx
from models.mlp import FingerspellingMLP

# ---------------------------------------------------------------------------
# Module-level fixture: export a single model once for all test runs
# ---------------------------------------------------------------------------

_INPUT_DIM = 63
_NUM_CLASSES = 36

# Build and export a model once at module load; reuse across all property runs.
_MODEL: FingerspellingMLP
_ONNX_PATH: Path
_SESSION: ort.InferenceSession


def _setup_model_and_session() -> tuple[FingerspellingMLP, Path, ort.InferenceSession]:
    """Create a FingerspellingMLP, export it to a temp ONNX file, load a session."""
    torch.manual_seed(42)
    model = FingerspellingMLP(
        input_dim=_INPUT_DIM,
        hidden_dims=(256, 256, 128),
        num_classes=_NUM_CLASSES,
        dropout=0.0,  # deterministic at eval time
    )
    model.eval()

    tmp_dir = tempfile.mkdtemp(prefix="onnx_roundtrip_")
    onnx_path = Path(tmp_dir) / "fingerspell.onnx"

    export_to_onnx(model, onnx_path, input_dim=_INPUT_DIM, opset=17)

    # Validate with ONNX checker (Requirement 7.1 / 7.3)
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    return model, onnx_path, session


_MODEL, _ONNX_PATH, _SESSION = _setup_model_and_session()


# ---------------------------------------------------------------------------
# Strategy: valid normalized landmark inputs
# ---------------------------------------------------------------------------

@st.composite
def landmark_batch(draw) -> np.ndarray:
    """
    Generate a batch of 1–8 landmark vectors.
    Each vector has _INPUT_DIM floats in the normalized range [-3.0, 3.0].
    """
    batch_size = draw(st.integers(min_value=1, max_value=8))
    flat = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=batch_size * _INPUT_DIM,
            max_size=batch_size * _INPUT_DIM,
        )
    )
    return np.array(flat, dtype=np.float32).reshape(batch_size, _INPUT_DIM)


# ---------------------------------------------------------------------------
# Property 12: ONNX model serialization round-trip
# **Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip**
# **Validates: Requirements 7.3**
# ---------------------------------------------------------------------------

@given(x=landmark_batch())
@settings(max_examples=100)
def test_onnx_round_trip_output_matches(x: np.ndarray) -> None:
    """
    **Feature: sign-language-interpreter, Property 12: ONNX model serialization round-trip**
    **Validates: Requirements 7.3**

    For any valid landmark input x, the ONNX session loaded from the saved file
    SHALL produce outputs identical (within atol=1e-5) to the original PyTorch
    model outputs.
    """
    # PyTorch reference output
    with torch.no_grad():
        pt_out = _MODEL(torch.from_numpy(x)).numpy()

    # ONNX Runtime output from the saved file
    input_name = _SESSION.get_inputs()[0].name
    ort_out = _SESSION.run(None, {input_name: x})[0]

    np.testing.assert_allclose(
        ort_out,
        pt_out,
        atol=1e-5,
        err_msg=(
            f"ONNX round-trip mismatch: max diff = "
            f"{float(np.max(np.abs(ort_out - pt_out))):.2e}"
        ),
    )
