"""
Property-based tests for landmark normalization and velocity features.

Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape
Validates: Requirements 3.1
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.landmarks import add_velocity_features, build_two_hand_vector, normalize_landmarks


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _hand_landmarks_array(draw):
    """Generate a valid (21, 3) hand landmark array."""
    # Values in [0, 1] like MediaPipe normalized coords
    flat = draw(st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=63, max_size=63))
    arr = np.array(flat, dtype=np.float32).reshape(21, 3)
    # Ensure wrist→index-MCP distance is non-degenerate
    arr[5] = arr[0] + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    return arr


@st.composite
def valid_hand_landmarks(draw):
    return _hand_landmarks_array(draw)


@st.composite
def valid_sequence(draw):
    """Generate a (T, 126) sequence of two-hand landmark vectors, T in [1, 60]."""
    T = draw(st.integers(min_value=1, max_value=60))
    # Use a fixed-size flat list capped at a small size to stay within entropy budget
    flat = draw(
        st.lists(
            st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
            min_size=126,
            max_size=126,
        )
    )
    # Tile the single-frame vector to fill T frames
    row = np.array(flat, dtype=np.float32)
    return np.tile(row, (T, 1))  # (T, 126)


# ---------------------------------------------------------------------------
# Property 4 — part A: normalize_landmarks output shape
# **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

@given(hand_lm=valid_hand_landmarks())
@settings(max_examples=100)
def test_normalize_landmarks_output_shape(hand_lm):
    """
    **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
    **Validates: Requirements 3.1**

    For any valid (21, 3) hand landmark array, normalize_landmarks() must return
    either None (degenerate scale) or a flat array of exactly 63 floats.
    """
    result = normalize_landmarks(hand_lm, mirror_left=False)
    if result is not None:
        assert result.shape == (63,), f"Expected (63,), got {result.shape}"
        assert result.dtype == np.float32


@given(hand_lm=valid_hand_landmarks())
@settings(max_examples=100)
def test_normalize_landmarks_mirrored_output_shape(hand_lm):
    """
    **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
    **Validates: Requirements 3.1**

    Mirroring (left-hand path) must produce the same shape as the right-hand path.
    """
    result = normalize_landmarks(hand_lm, mirror_left=True)
    if result is not None:
        assert result.shape == (63,), f"Expected (63,), got {result.shape}"


@given(
    right_lm=valid_hand_landmarks(),
    left_lm=valid_hand_landmarks(),
)
@settings(max_examples=100)
def test_build_two_hand_vector_shape(right_lm, left_lm):
    """
    **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
    **Validates: Requirements 3.1**

    build_two_hand_vector() must always return a 126-float array regardless of
    which hands are present.
    """
    for r, l in [(right_lm, left_lm), (right_lm, None), (None, left_lm), (None, None)]:
        vec = build_two_hand_vector(r, l)
        assert vec.shape == (126,), f"Expected (126,), got {vec.shape}"
        assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# Property 4 — part B: add_velocity_features output shape
# **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

@given(seq=valid_sequence())
@settings(max_examples=100)
def test_add_velocity_features_shape(seq):
    """
    **Feature: sign-language-interpreter, Property 4: Word-level feature vector has correct shape**
    **Validates: Requirements 3.1**

    For any (T, 126) sequence, add_velocity_features() must return shape (T, 252)
    with the first 126 columns equal to the original positions and the last 126
    being frame-over-frame deltas (zero-padded at t=0).
    """
    result = add_velocity_features(seq)

    T = seq.shape[0]
    assert result.shape == (T, 252), f"Expected ({T}, 252), got {result.shape}"

    # First 126 columns are original positions
    np.testing.assert_array_equal(result[:, :126], seq.astype(np.float32))

    # Frame 0 velocity must be zeros (no prior frame)
    np.testing.assert_array_equal(result[0, 126:], np.zeros(126, dtype=np.float32))

    # Remaining velocity frames must be seq[t] - seq[t-1]
    if T > 1:
        expected_velocity = seq[1:].astype(np.float32) - seq[:-1].astype(np.float32)
        np.testing.assert_allclose(result[1:, 126:], expected_velocity, atol=1e-6)
