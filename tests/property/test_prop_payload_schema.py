"""
Property-based tests for WebSocket payload validation and response schema.

Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays
Validates: Requirements 8.2

Feature: sign-language-interpreter, Property 15: WebSocket response contains all required fields
Validates: Requirements 8.3
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.testclient import TestClient

from server.main import app, validate_fingerspell_payload, validate_word_payload

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

FINGERSPELL_DIM = 63
WORD_DIM = 126


@st.composite
def landmarks_wrong_length_for_fingerspell(draw):
    """Generate a list of floats whose length is != 63."""
    # Pick any length except 63
    length = draw(st.integers(min_value=0, max_value=200).filter(lambda n: n != FINGERSPELL_DIM))
    values = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=length,
            max_size=length,
        )
    )
    return values


@st.composite
def landmarks_correct_length_for_fingerspell(draw):
    """Generate a list of exactly 63 floats."""
    values = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=FINGERSPELL_DIM,
            max_size=FINGERSPELL_DIM,
        )
    )
    return values


@st.composite
def landmarks_wrong_length_for_word(draw):
    """Generate a list of floats whose length is != 126."""
    length = draw(st.integers(min_value=0, max_value=300).filter(lambda n: n != WORD_DIM))
    values = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=length,
            max_size=length,
        )
    )
    return values


@st.composite
def landmarks_correct_length_for_word(draw):
    """Generate a list of exactly 126 floats."""
    values = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=WORD_DIM,
            max_size=WORD_DIM,
        )
    )
    return values


# ---------------------------------------------------------------------------
# Property 14 — WebSocket payload validation rejects wrong-length arrays
# **Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays**
# **Validates: Requirements 8.2**
# ---------------------------------------------------------------------------


@given(landmarks=landmarks_wrong_length_for_fingerspell())
@settings(max_examples=100)
def test_fingerspell_rejects_wrong_length(landmarks):
    """
    **Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays**
    **Validates: Requirements 8.2**

    For any landmark list whose length != 63, validate_fingerspell_payload()
    SHALL return a structured error response (non-None dict with type='error').
    """
    payload = {"landmarks": landmarks}
    result = validate_fingerspell_payload(payload)

    assert result is not None, (
        f"Expected a validation error for landmarks of length {len(landmarks)}, "
        f"but validation passed"
    )
    assert isinstance(result, dict), "Error response must be a dict"
    assert result.get("type") == "error", (
        f"Error response must have type='error', got {result.get('type')!r}"
    )
    assert "message" in result, "Error response must include a 'message' field"
    assert isinstance(result["message"], str) and len(result["message"]) > 0, (
        "Error message must be a non-empty string"
    )


@given(landmarks=landmarks_correct_length_for_fingerspell())
@settings(max_examples=100)
def test_fingerspell_accepts_correct_length(landmarks):
    """
    **Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays**
    **Validates: Requirements 8.2**

    For any landmark list of exactly 63 floats, validate_fingerspell_payload()
    SHALL return None (indicating the payload is valid and ready for inference).
    """
    payload = {"landmarks": landmarks}
    result = validate_fingerspell_payload(payload)

    assert result is None, (
        f"Expected None (valid payload) for landmarks of length {len(landmarks)}, "
        f"but got error: {result}"
    )


@given(landmarks=landmarks_wrong_length_for_word())
@settings(max_examples=100)
def test_word_rejects_wrong_length(landmarks):
    """
    **Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays**
    **Validates: Requirements 8.2**

    For any landmark list whose length != 126, validate_word_payload()
    SHALL return a structured error response (non-None dict with type='error').
    """
    payload = {"landmarks": landmarks}
    result = validate_word_payload(payload)

    assert result is not None, (
        f"Expected a validation error for landmarks of length {len(landmarks)}, "
        f"but validation passed"
    )
    assert isinstance(result, dict), "Error response must be a dict"
    assert result.get("type") == "error", (
        f"Error response must have type='error', got {result.get('type')!r}"
    )
    assert "message" in result, "Error response must include a 'message' field"
    assert isinstance(result["message"], str) and len(result["message"]) > 0, (
        "Error message must be a non-empty string"
    )


@given(landmarks=landmarks_correct_length_for_word())
@settings(max_examples=100)
def test_word_accepts_correct_length(landmarks):
    """
    **Feature: sign-language-interpreter, Property 14: WebSocket payload validation rejects wrong-length arrays**
    **Validates: Requirements 8.2**

    For any landmark list of exactly 126 floats, validate_word_payload()
    SHALL return None (indicating the payload is valid and ready for inference).
    """
    payload = {"landmarks": landmarks}
    result = validate_word_payload(payload)

    assert result is None, (
        f"Expected None (valid payload) for landmarks of length {len(landmarks)}, "
        f"but got error: {result}"
    )


# ---------------------------------------------------------------------------
# Additional edge-case coverage: missing key and non-list type
# ---------------------------------------------------------------------------


def test_fingerspell_rejects_missing_landmarks_key():
    """Missing 'landmarks' key must return an error."""
    result = validate_fingerspell_payload({"handedness": "Right"})
    assert result is not None
    assert result.get("type") == "error"


def test_word_rejects_missing_landmarks_key():
    """Missing 'landmarks' key must return an error."""
    result = validate_word_payload({"frame_idx": 0})
    assert result is not None
    assert result.get("type") == "error"


def test_fingerspell_rejects_non_list_landmarks():
    """Non-list landmarks value must return an error."""
    result = validate_fingerspell_payload({"landmarks": "not a list"})
    assert result is not None
    assert result.get("type") == "error"


def test_word_rejects_non_list_landmarks():
    """Non-list landmarks value must return an error."""
    result = validate_word_payload({"landmarks": 12345})
    assert result is not None
    assert result.get("type") == "error"


# ---------------------------------------------------------------------------
# Shared required-fields constants (Property 15)
# ---------------------------------------------------------------------------

# Requirement 8.3 mandates all of these fields in every successful inference response.
REQUIRED_RESPONSE_FIELDS = {"type", "prediction", "confidence", "top5", "accepted", "latency_ms"}

FINGERSPELL_DIM = 63
WORD_DIM = 126


# ---------------------------------------------------------------------------
# Generators for Property 15
# ---------------------------------------------------------------------------


@st.composite
def valid_fingerspell_landmarks(draw):
    """Generate a list of exactly 63 finite floats in [-3.0, 3.0]."""
    return draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=FINGERSPELL_DIM,
            max_size=FINGERSPELL_DIM,
        )
    )


@st.composite
def valid_word_landmarks(draw):
    """Generate a list of exactly 126 finite floats in [-3.0, 3.0]."""
    return draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=WORD_DIM,
            max_size=WORD_DIM,
        )
    )


# ---------------------------------------------------------------------------
# Property 15 — WebSocket response contains all required fields
# **Feature: sign-language-interpreter, Property 15: WebSocket response contains all required fields**
# **Validates: Requirements 8.3**
# ---------------------------------------------------------------------------


@given(landmarks=valid_fingerspell_landmarks())
@settings(max_examples=100)
def test_fingerspell_response_contains_all_required_fields(landmarks):
    """
    **Feature: sign-language-interpreter, Property 15: WebSocket response contains all required fields**
    **Validates: Requirements 8.3**

    For any valid 63-float landmark input to /ws/fingerspell, the JSON response
    SHALL contain all of: type, prediction, confidence, top5, accepted, latency_ms.
    No required field SHALL be absent from any successful inference response.
    """
    import server.main as main_module

    n_classes = 26
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    uniform_probs = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)

    # Mock _run_fingerspell_inference to return a uniform probability array
    # without touching ONNX at all.  run_in_executor is left untouched so
    # TestClient's anyio backend can call it normally.
    with (
        patch.object(main_module, "_fingerspell_session", MagicMock()),
        patch.object(main_module, "_fingerspell_labels", labels),
        patch.object(main_module, "_run_fingerspell_inference", return_value=uniform_probs),
    ):
        with TestClient(app) as client:
            with client.websocket_connect("/ws/fingerspell") as ws:
                ws.send_text(json.dumps({"landmarks": landmarks, "handedness": "Right"}))
                raw = ws.receive_text()

    response = json.loads(raw)

    missing = REQUIRED_RESPONSE_FIELDS - set(response.keys())
    assert not missing, (
        f"Response is missing required fields: {missing}. "
        f"Got keys: {set(response.keys())}"
    )

    # Additional type checks to ensure the fields carry the right data types.
    assert isinstance(response["type"], str), "'type' must be a string"
    assert isinstance(response["prediction"], str), "'prediction' must be a string"
    assert isinstance(response["confidence"], (int, float)), "'confidence' must be numeric"
    assert isinstance(response["top5"], list), "'top5' must be a list"
    assert isinstance(response["accepted"], bool), "'accepted' must be a bool"
    assert isinstance(response["latency_ms"], (int, float)), "'latency_ms' must be numeric"

    # Confidence must be a valid probability in [0.0, 1.0].
    assert 0.0 <= response["confidence"] <= 1.0, (
        f"'confidence' must be in [0.0, 1.0], got {response['confidence']}"
    )

    # top5 entries must each have 'label' and 'prob' keys.
    for entry in response["top5"]:
        assert "label" in entry and "prob" in entry, (
            f"Each top5 entry must have 'label' and 'prob', got {entry}"
        )


@given(landmarks=valid_word_landmarks())
@settings(max_examples=100)
def test_word_response_contains_all_required_fields(landmarks):
    """
    **Feature: sign-language-interpreter, Property 15: WebSocket response contains all required fields**
    **Validates: Requirements 8.3**

    For any valid 126-float landmark input to /ws/word (sent enough times to
    trigger inference via word_stride), the JSON response SHALL contain all of:
    type, prediction, confidence, top5, accepted, latency_ms.
    No required field SHALL be absent from any successful inference response.
    """
    import server.main as main_module

    n_classes = 26
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    uniform_probs = np.full(n_classes, 1.0 / n_classes, dtype=np.float32)

    # Mock _run_word_inference to return a uniform probability array.
    with (
        patch.object(main_module, "_word_session", MagicMock()),
        patch.object(main_module, "_word_labels", labels),
        patch.object(main_module, "_run_word_inference", return_value=uniform_probs),
    ):
        word_stride = main_module.settings.word_stride
        with TestClient(app) as client:
            with client.websocket_connect("/ws/word") as ws:
                # Send word_stride frames to trigger one inference cycle.
                response_text = None
                for frame_idx in range(word_stride):
                    ws.send_text(
                        json.dumps({"landmarks": landmarks, "frame_idx": frame_idx})
                    )
                    # The stride-th frame triggers inference and returns a response.
                    if frame_idx == word_stride - 1:
                        response_text = ws.receive_text()

    assert response_text is not None, (
        f"Expected a response after {word_stride} frames (word_stride={word_stride}), "
        "but no message was received"
    )

    response = json.loads(response_text)

    missing = REQUIRED_RESPONSE_FIELDS - set(response.keys())
    assert not missing, (
        f"Response is missing required fields: {missing}. "
        f"Got keys: {set(response.keys())}"
    )

    # Additional type checks.
    assert isinstance(response["type"], str), "'type' must be a string"
    assert isinstance(response["prediction"], str), "'prediction' must be a string"
    assert isinstance(response["confidence"], (int, float)), "'confidence' must be numeric"
    assert isinstance(response["top5"], list), "'top5' must be a list"
    assert isinstance(response["accepted"], bool), "'accepted' must be a bool"
    assert isinstance(response["latency_ms"], (int, float)), "'latency_ms' must be numeric"

    assert 0.0 <= response["confidence"] <= 1.0, (
        f"'confidence' must be in [0.0, 1.0], got {response['confidence']}"
    )

    for entry in response["top5"]:
        assert "label" in entry and "prob" in entry, (
            f"Each top5 entry must have 'label' and 'prob', got {entry}"
        )


# ---------------------------------------------------------------------------
# Property 16 — Landmark JSON serialization round-trip
# # Feature: sign-language-interpreter, Property 16: Landmark JSON serialization round-trip
# **Validates: Requirements 8.5**
# ---------------------------------------------------------------------------


@st.composite
def landmark_array_normalized(draw):
    """
    Generate a float list of length 63 (fingerspell) or 126 (word-level)
    with values in the normalized landmark range [-1.0, 1.0].

    Requirement 8.5 specifies the server validates values in [-1.0, 1.0].
    """
    length = draw(st.sampled_from([FINGERSPELL_DIM, WORD_DIM]))
    values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=length,
            max_size=length,
        )
    )
    return values


@given(landmarks=landmark_array_normalized())
@settings(max_examples=100)
def test_landmark_json_round_trip(landmarks):
    """
    **Feature: sign-language-interpreter, Property 16: Landmark JSON serialization round-trip**
    **Validates: Requirements 8.5**

    For any landmark array with values in the expected normalized range [-1.0, 1.0],
    serializing to JSON and deserializing back SHALL preserve all values to within
    floating-point precision (absolute error ≤ 1e-6).
    """
    # Serialize to JSON and deserialize back
    serialized = json.dumps(landmarks)
    deserialized = json.loads(serialized)

    assert len(deserialized) == len(landmarks), (
        f"Round-trip changed array length: expected {len(landmarks)}, got {len(deserialized)}"
    )

    original = np.array(landmarks, dtype=np.float64)
    recovered = np.array(deserialized, dtype=np.float64)

    np.testing.assert_allclose(
        recovered,
        original,
        atol=1e-6,
        err_msg=(
            f"Landmark JSON round-trip exceeded absolute tolerance of 1e-6. "
            f"Max error: {np.max(np.abs(recovered - original))}"
        ),
    )
