"""Unit tests for ONNX export pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export_onnx import export_to_onnx, quantize_to_int8
from models.mlp import FingerspellingMLP


def _make_model() -> FingerspellingMLP:
    model = FingerspellingMLP(
        input_dim=63,
        hidden_dims=(256, 256, 128),
        num_classes=36,
        dropout=0.0,
    )
    model.eval()
    return model


class TestOnnxExport:
    def test_export_produces_valid_onnx_file(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            export_to_onnx(model, path, input_dim=63)
            assert path.exists()
            proto = onnx.load(str(path))
            onnx.checker.check_model(proto)

    def test_export_has_correct_io_names_and_shapes(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            export_to_onnx(model, path, input_dim=63)

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            outputs = session.get_outputs()

            assert inputs[0].name == "landmarks"
            assert outputs[0].name == "logits"
            assert inputs[0].shape[1] == 63
            assert outputs[0].shape[1] == 36

    def test_export_output_matches_pytorch(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.onnx"
            export_to_onnx(model, path, input_dim=63)

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            x = np.random.randn(4, 63).astype(np.float32)

            with torch.no_grad():
                pt_out = model(torch.from_numpy(x)).numpy()

            ort_out = session.run(None, {session.get_inputs()[0].name: x})[0]
            np.testing.assert_allclose(ort_out, pt_out, atol=1e-4)

    def test_int8_quantization_produces_loadable_model(self):
        model = _make_model()
        with tempfile.TemporaryDirectory() as tmp:
            fp32_path = Path(tmp) / "model.onnx"
            int8_path = Path(tmp) / "model_int8.onnx"
            export_to_onnx(model, fp32_path, input_dim=63)
            quantize_to_int8(fp32_path, int8_path)

            assert int8_path.exists()
            session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
            x = np.random.randn(2, 63).astype(np.float32)
            out = session.run(None, {session.get_inputs()[0].name: x})[0]
            assert out.shape == (2, 36)
