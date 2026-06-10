"""Unit tests for landmark normalization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from utils.landmarks import (
    add_velocity_features,
    build_two_hand_vector,
    normalize_landmarks,
)


def _valid_hand() -> np.ndarray:
    """Hand landmarks with a non-degenerate wrist→index-MCP distance."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.5, 0.0]   # wrist
    lm[5] = [0.6, 0.5, 0.0]   # index MCP — 0.1 units from wrist
    lm[8] = [0.55, 0.4, 0.0]  # middle tip
    return lm


class TestNormalizeLandmarks:
    def test_wrist_translated_to_origin(self):
        lm = _valid_hand()
        result = normalize_landmarks(lm)
        assert result is not None
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] == pytest.approx(0.0, abs=1e-5)

    def test_output_shape_is_63(self):
        result = normalize_landmarks(_valid_hand())
        assert result is not None
        assert result.shape == (63,)

    def test_degenerate_hand_returns_none(self):
        zeros = np.zeros((21, 3), dtype=np.float32)
        assert normalize_landmarks(zeros) is None

    def test_mirror_left_flips_x_axis(self):
        lm = _valid_hand()
        right = normalize_landmarks(lm, mirror_left=False)
        left = normalize_landmarks(lm, mirror_left=True)
        assert right is not None and left is not None
        # x-coordinates (indices 0, 3, 6, …) should be negated when mirrored
        for i in range(0, 63, 3):
            assert left[i] == pytest.approx(-right[i], abs=1e-5)

    def test_invalid_shape_returns_none(self):
        assert normalize_landmarks(np.zeros((20, 3))) is None


class TestBuildTwoHandVector:
    def test_both_hands_produces_126_floats(self):
        hand = _valid_hand()
        vec = build_two_hand_vector(hand, hand)
        assert vec.shape == (126,)

    def test_missing_hands_are_zero_padded(self):
        hand = _valid_hand()
        vec = build_two_hand_vector(hand, None)
        assert vec.shape == (126,)
        assert np.all(vec[63:] == 0)

    def test_both_none_returns_all_zeros(self):
        vec = build_two_hand_vector(None, None)
        assert vec.shape == (126,)
        assert np.all(vec == 0)


class TestAddVelocityFeatures:
    def test_output_shape_doubles_features(self):
        seq = np.random.randn(10, 126).astype(np.float32)
        out = add_velocity_features(seq)
        assert out.shape == (10, 252)

    def test_first_frame_velocity_is_zero(self):
        seq = np.ones((5, 126), dtype=np.float32)
        out = add_velocity_features(seq)
        assert np.all(out[0, 126:] == 0)

    def test_velocity_matches_frame_delta(self):
        seq = np.array([[1.0] * 126, [3.0] * 126], dtype=np.float32)
        out = add_velocity_features(seq)
        np.testing.assert_allclose(out[1, 126:], 2.0, rtol=1e-5)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="Expected input shape"):
            add_velocity_features(np.zeros((10, 64)))
