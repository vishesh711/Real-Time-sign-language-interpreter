"""
FastAPI WebSocket server for the Real-Time Sign Language Interpreter.

Exposes three HTTP endpoints and two WebSocket endpoints:
  GET  /health            — liveness probe
  GET  /info              — model metadata
  WS   /ws/fingerspell    — single-frame landmark → letter
  WS   /ws/word           — rolling frame-buffer → gloss word

All ONNX sessions are loaded once at startup via a lifespan context manager.
A shared ThreadPoolExecutor handles the CPU-bound word-inference calls so the
async event loop is never blocked.

Requirements: 8.1, 8.2, 8.3, 8.4, 3.3
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from server.config import Settings
from utils.gate import PredictionGate
from utils.landmarks import IDX_TO_LABEL, add_velocity_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (populated during lifespan)
# ---------------------------------------------------------------------------

settings = Settings()

_fingerspell_session = None   # onnxruntime.InferenceSession
_word_session = None          # onnxruntime.InferenceSession
_fingerspell_labels: List[str] = []
_word_labels: List[str] = []
_thread_pool: Optional[ThreadPoolExecutor] = None

# Model metadata (set at startup)
_fingerspell_meta: Dict[str, Any] = {}
_word_meta: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_providers() -> List[str]:
    """Return ORT execution providers in priority order: CUDA → CoreML → DirectML → CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        available = []

    priority = ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                "DmlExecutionProvider", "CPUExecutionProvider"]
    selected = [p for p in priority if p in available]
    # Always ensure CPU is a fallback
    if "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")
    return selected


def _make_session_options():
    """Create ORT SessionOptions with sequential mode, 1 intra-op thread, full graph optimisation."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _load_session(path: str):
    """Load an ONNX InferenceSession from *path* with the selected providers."""
    import onnxruntime as ort

    providers = _get_providers()
    opts = _make_session_options()
    session = ort.InferenceSession(path, sess_options=opts, providers=providers)
    return session


def _load_labels(json_path: str, fallback: Optional[List[str]] = None) -> List[str]:
    """Load labels from a JSON sidecar file, falling back to *fallback* on error."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and all(isinstance(l, str) for l in labels):
            return labels
    except Exception as exc:
        logger.warning("Could not load labels from %s: %s", json_path, exc)
    if fallback is not None:
        return fallback
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with optional temperature scaling."""
    scaled = logits.astype(np.float32) / max(temperature, 1e-8)
    shifted = scaled - scaled.max()
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def _run_fingerspell_inference(landmarks: List[float]) -> Dict[str, Any]:
    """Run the fingerspell ONNX model synchronously (intended for thread pool)."""
    x = np.array(landmarks, dtype=np.float32).reshape(1, 63)
    input_name = _fingerspell_session.get_inputs()[0].name
    outputs = _fingerspell_session.run(None, {input_name: x})
    logits = outputs[0][0]  # shape: (N_CLASSES,)
    probs = _softmax(logits, temperature=settings.temperature)
    return probs


def _run_word_inference(frames: np.ndarray) -> Dict[str, Any]:
    """Run the word ONNX model synchronously (intended for thread pool).

    Args:
        frames: Array of shape (seq_len, 252) — resampled + velocity features.

    Returns:
        Probability array of shape (N_CLASSES,).
    """
    x = frames.astype(np.float32)[np.newaxis]  # (1, seq_len, 252)
    input_name = _word_session.get_inputs()[0].name
    outputs = _word_session.run(None, {input_name: x})
    logits = outputs[0][0]  # shape: (N_CLASSES,)
    probs = _softmax(logits, temperature=settings.temperature)
    return probs


def _resample_sequence(frames: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate *frames* to exactly *target_len* time steps.

    Args:
        frames:     Array of shape (T, D).
        target_len: Desired number of frames.

    Returns:
        Array of shape (target_len, D).
    """
    T, D = frames.shape
    if T == target_len:
        return frames.copy()
    src_indices = np.linspace(0, T - 1, num=target_len)
    lo = np.floor(src_indices).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (src_indices - lo)[:, np.newaxis]  # (target_len, 1)
    resampled = frames[lo] * (1.0 - frac) + frames[hi] * frac
    return resampled.astype(np.float32)


def _build_top5(probs: np.ndarray, labels: List[str]) -> List[Dict[str, Any]]:
    """Return the top-5 (label, prob) pairs sorted by descending probability."""
    n = min(5, len(probs))
    top_indices = np.argsort(probs)[::-1][:n]
    return [
        {"label": labels[i] if i < len(labels) else str(i), "prob": float(probs[i])}
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# Payload validation helpers (Requirements 8.2)
# ---------------------------------------------------------------------------

def validate_fingerspell_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Validate a fingerspell WebSocket payload.

    Returns None if the payload is valid (landmarks is a list of exactly 63
    numeric values), or a structured error dict if it is invalid.

    Args:
        payload: Parsed JSON object (expected to be a dict).

    Returns:
        None on success, or {"type": "error", "message": str} on failure.
    """
    if not isinstance(payload, dict):
        return {"type": "error", "message": "Payload must be a JSON object"}

    if "landmarks" not in payload:
        return {"type": "error", "message": "Missing 'landmarks' key"}

    landmarks = payload["landmarks"]
    if not isinstance(landmarks, list) or len(landmarks) != 63:
        length_info = (
            len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__
        )
        return {
            "type": "error",
            "message": (
                f"'landmarks' must be a list of exactly 63 floats, got {length_info}"
            ),
        }

    try:
        [float(v) for v in landmarks]
    except (TypeError, ValueError):
        return {"type": "error", "message": "'landmarks' must contain numeric values"}

    return None


def validate_word_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Validate a word WebSocket payload.

    Returns None if the payload is valid (landmarks is a list of exactly 126
    numeric values), or a structured error dict if it is invalid.

    Args:
        payload: Parsed JSON object (expected to be a dict).

    Returns:
        None on success, or {"type": "error", "message": str} on failure.
    """
    if not isinstance(payload, dict):
        return {"type": "error", "message": "Payload must be a JSON object"}

    if "landmarks" not in payload:
        return {"type": "error", "message": "Missing 'landmarks' key"}

    landmarks = payload["landmarks"]
    if not isinstance(landmarks, list) or len(landmarks) != 126:
        length_info = (
            len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__
        )
        return {
            "type": "error",
            "message": (
                f"'landmarks' must be a list of exactly 126 floats, got {length_info}"
            ),
        }

    try:
        [float(v) for v in landmarks]
    except (TypeError, ValueError):
        return {"type": "error", "message": "'landmarks' must contain numeric values"}

    return None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX sessions and thread pool at startup; clean up on shutdown."""
    global _fingerspell_session, _word_session
    global _fingerspell_labels, _word_labels
    global _thread_pool, _fingerspell_meta, _word_meta

    _thread_pool = ThreadPoolExecutor(max_workers=2)

    # --- Fingerspell session ---
    fs_path = settings.fingerspell_onnx_path
    if os.path.isfile(fs_path):
        try:
            _fingerspell_session = _load_session(fs_path)
            # Infer vocab size from model output shape
            out_shape = _fingerspell_session.get_outputs()[0].shape
            n_classes = out_shape[1] if len(out_shape) > 1 and isinstance(out_shape[1], int) else None
            logger.info("Loaded fingerspell session from %s (n_classes=%s)", fs_path, n_classes)
        except Exception as exc:
            logger.error("Failed to load fingerspell session: %s", exc)
    else:
        logger.warning("Fingerspell ONNX model not found at %s — endpoint unavailable", fs_path)

    # --- Word session ---
    wd_path = settings.word_onnx_path
    if os.path.isfile(wd_path):
        try:
            _word_session = _load_session(wd_path)
            out_shape = _word_session.get_outputs()[0].shape
            n_classes = out_shape[1] if len(out_shape) > 1 and isinstance(out_shape[1], int) else None
            logger.info("Loaded word session from %s (n_classes=%s)", wd_path, n_classes)
        except Exception as exc:
            logger.error("Failed to load word session: %s", exc)
    else:
        logger.warning("Word ONNX model not found at %s — endpoint unavailable", wd_path)

    # --- Labels ---
    fallback_fs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    _fingerspell_labels = _load_labels(settings.labels_json_path, fallback=fallback_fs)
    _word_labels = _load_labels(settings.labels_json_path, fallback=fallback_fs)

    # --- Build metadata dicts ---
    providers = _get_providers()
    _fingerspell_meta = {
        "vocab_size": len(_fingerspell_labels),
        "feature_dim": 63,
        "available_providers": providers,
        "model_loaded": _fingerspell_session is not None,
    }
    _word_meta = {
        "vocab_size": len(_word_labels),
        "seq_len": settings.seq_len,
        "feature_dim": 252,
        "available_providers": providers,
        "model_loaded": _word_session is not None,
    }

    yield

    # Shutdown
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Sign Language Interpreter", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/info")
async def info():
    """Return model metadata for both endpoints."""
    return {
        "fingerspell": _fingerspell_meta,
        "word": _word_meta,
    }


# ---------------------------------------------------------------------------
# WebSocket: /ws/fingerspell  (Task 7.2)
# ---------------------------------------------------------------------------

@app.websocket("/ws/fingerspell")
async def ws_fingerspell(websocket: WebSocket):
    """Single-frame landmark → letter prediction.

    Client sends:
        {"landmarks": [63 floats], "handedness": "Right"}

    Server responds:
        {
          "type": "fingerspell",
          "prediction": "H",
          "confidence": 0.94,
          "top5": [{"label": "H", "prob": 0.94}, ...],
          "accepted": true,
          "accepted_label": "H",
          "latency_ms": 3.2
        }
    """
    await websocket.accept()

    # Per-connection PredictionGate
    gate = PredictionGate(
        vote_window=settings.vote_window,
        hold_frames=settings.hold_frames,
        confidence_threshold=settings.classifier_threshold,
        cooldown_frames=settings.cooldown_frames,
    )

    try:
        while True:
            t_start = time.perf_counter()

            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON payload"}
                )
                continue

            # --- Validate payload ---
            if "landmarks" not in payload:
                await websocket.send_json(
                    {"type": "error", "message": "Missing 'landmarks' key"}
                )
                continue

            landmarks = payload["landmarks"]
            if not isinstance(landmarks, list) or len(landmarks) != 63:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            f"'landmarks' must be a list of exactly 63 floats, "
                            f"got {len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__}"
                        ),
                    }
                )
                continue

            try:
                landmarks = [float(v) for v in landmarks]
            except (TypeError, ValueError):
                await websocket.send_json(
                    {"type": "error", "message": "'landmarks' must contain numeric values"}
                )
                continue

            # Handedness is optional
            handedness = payload.get("handedness", "Right")
            if handedness not in ("Left", "Right"):
                handedness = "Right"

            # --- Check model availability ---
            if _fingerspell_session is None:
                await websocket.send_json(
                    {"type": "error", "message": "Fingerspell model not loaded"}
                )
                continue

            # --- Run inference in thread pool (non-blocking) ---
            loop = asyncio.get_event_loop()
            probs = await loop.run_in_executor(
                _thread_pool, _run_fingerspell_inference, landmarks
            )

            # --- Build response ---
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            prediction = (
                _fingerspell_labels[top_idx]
                if top_idx < len(_fingerspell_labels)
                else str(top_idx)
            )
            top5 = _build_top5(probs, _fingerspell_labels)

            # --- PredictionGate ---
            accepted_label = gate.update(prediction, confidence)
            accepted = accepted_label is not None

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            await websocket.send_json(
                {
                    "type": "fingerspell",
                    "prediction": prediction,
                    "confidence": confidence,
                    "top5": top5,
                    "accepted": accepted,
                    "accepted_label": accepted_label,
                    "latency_ms": round(latency_ms, 3),
                }
            )

    except WebSocketDisconnect:
        logger.info("Fingerspell client disconnected")
    except Exception as exc:
        logger.exception("Error in /ws/fingerspell: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: /ws/word  (Task 7.5)
# ---------------------------------------------------------------------------

@app.websocket("/ws/word")
async def ws_word(websocket: WebSocket):
    """Rolling frame-buffer stream → word-level gloss prediction.

    Client sends one of:
        {"landmarks": [126 floats], "frame_idx": N}
        {"type": "clear"}

    Server responds (every word_stride frames):
        {
          "type": "word",
          "prediction": "HELLO",
          "confidence": 0.87,
          "top5": [...],
          "accepted": true,
          "accepted_label": "HELLO",
          "latency_ms": 12.4
        }
    """
    await websocket.accept()

    seq_len = settings.seq_len
    max_buffer = 2 * seq_len  # cap at 2 × seq_len
    frame_count = 0  # frames since last inference

    # Per-connection rolling buffer: list of 126-float arrays
    frame_buffer: deque[np.ndarray] = deque(maxlen=max_buffer)

    # Per-connection PredictionGate
    gate = PredictionGate(
        vote_window=settings.vote_window,
        hold_frames=settings.hold_frames,
        confidence_threshold=settings.classifier_threshold,
        cooldown_frames=settings.cooldown_frames,
    )

    try:
        while True:
            t_start = time.perf_counter()

            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON payload"}
                )
                continue

            # --- Clear command ---
            if payload.get("type") == "clear":
                frame_buffer.clear()
                gate.reset()
                frame_count = 0
                continue

            # --- Validate frame payload ---
            if "landmarks" not in payload:
                await websocket.send_json(
                    {"type": "error", "message": "Missing 'landmarks' key"}
                )
                continue

            landmarks = payload["landmarks"]
            if not isinstance(landmarks, list) or len(landmarks) != 126:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            f"'landmarks' must be a list of exactly 126 floats, "
                            f"got {len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__}"
                        ),
                    }
                )
                continue

            try:
                landmarks = [float(v) for v in landmarks]
            except (TypeError, ValueError):
                await websocket.send_json(
                    {"type": "error", "message": "'landmarks' must contain numeric values"}
                )
                continue

            # --- Append to rolling buffer ---
            frame_buffer.append(np.array(landmarks, dtype=np.float32))
            frame_count += 1

            # --- Run inference every word_stride frames ---
            if frame_count % settings.word_stride != 0:
                continue

            if _word_session is None:
                await websocket.send_json(
                    {"type": "error", "message": "Word model not loaded"}
                )
                continue

            if len(frame_buffer) < 1:
                continue

            # Convert buffer → (T, 126) array
            buf_array = np.stack(list(frame_buffer), axis=0)  # (T, 126)

            # Resample to exactly seq_len
            resampled = _resample_sequence(buf_array, seq_len)  # (seq_len, 126)

            # Append velocity features → (seq_len, 252)
            with_velocity = add_velocity_features(resampled)  # (seq_len, 252)

            # Run inference in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            probs = await loop.run_in_executor(
                _thread_pool, _run_word_inference, with_velocity
            )

            # --- Build response ---
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            prediction = (
                _word_labels[top_idx]
                if top_idx < len(_word_labels)
                else str(top_idx)
            )
            top5 = _build_top5(probs, _word_labels)

            # --- PredictionGate ---
            accepted_label = gate.update(prediction, confidence)
            accepted = accepted_label is not None

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            await websocket.send_json(
                {
                    "type": "word",
                    "prediction": prediction,
                    "confidence": confidence,
                    "top5": top5,
                    "accepted": accepted,
                    "accepted_label": accepted_label,
                    "latency_ms": round(latency_ms, 3),
                }
            )

    except WebSocketDisconnect:
        logger.info("Word client disconnected")
    except Exception as exc:
        logger.exception("Error in /ws/word: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
