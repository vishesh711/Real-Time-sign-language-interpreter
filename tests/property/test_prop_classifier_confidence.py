"""
Property-based tests for classifier confidence range.

# Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
**Validates: Requirements 2.2, 3.2**

Property 2 — Classifier confidence is always a valid probability:
    For any normalized landmark input to either the FingerspellingMLP or the
    SignLSTM model, the returned confidence score SHALL be in the range
    [0.0, 1.0] inclusive.

    The confidence score is defined as max(softmax(logits)), i.e. the highest
    probability in the softmax output distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

from models.mlp import FingerspellingMLP
from models.cnn3d_lstm import SignLSTM

# ---------------------------------------------------------------------------
# Module-level model instances (created once; no trained checkpoint needed)
# ---------------------------------------------------------------------------

# FingerspellingMLP with default args: input_dim=63, 26 output classes
_FINGERSPELL_MODEL = FingerspellingMLP(
    input_dim=63,
    hidden_dims=(256, 256, 128),
    num_classes=26,
    dropout=0.0,  # disable dropout for deterministic eval
)
_FINGERSPELL_MODEL.eval()

# SignLSTM with small config for faster test execution; still validates the
# full forward-pass → softmax → max confidence pipeline.
_SIGN_LSTM_MODEL = SignLSTM(
    seq_len=30,
    feature_dim=252,
    num_classes=50,   # reduced from 300 to keep tests fast
    cnn_channels=(8, 16, 32),
    lstm_hidden=64,
    lstm_layers=2,
    lstm_dropout=0.0,
    dropout=0.0,
)
_SIGN_LSTM_MODEL.eval()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def fingerspell_input(draw) -> torch.Tensor:
    """Generate a batch-1 float32 tensor of shape [1, 63] for FingerspellingMLP.

    Values are drawn from a plausible normalized range [-3.0, 3.0] matching
    wrist-relative normalized landmark coordinates.
    """
    raw = draw(
        st.lists(
            st.floats(
                min_value=-3.0,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=63,
            max_size=63,
        )
    )
    arr = np.array(raw, dtype=np.float32).reshape(1, 63)
    return torch.from_numpy(arr)


@st.composite
def word_level_input(draw) -> torch.Tensor:
    """Generate a batch-1 float32 tensor of shape [1, 30, 252] for SignLSTM.

    Uses hypothesis.extra.numpy to generate the full array in one shot, which
    keeps the base-example size manageable for Hypothesis's internal shrinking.
    Values are drawn from the normalized range [-3.0, 3.0].
    """
    arr = draw(
        np_arrays(
            dtype=np.float32,
            shape=(1, 30, 252),
            elements=st.floats(
                min_value=-3.0,
                max_value=3.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Property 2a — FingerspellingMLP confidence in [0.0, 1.0]
# **Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability**
# **Validates: Requirements 2.2**
# ---------------------------------------------------------------------------

@given(x=fingerspell_input())
@settings(max_examples=100)
def test_fingerspell_mlp_confidence_is_valid_probability(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 2.2**

    For any normalized float32 input of shape [1, 63], applying softmax to
    FingerspellingMLP's output logits and taking the maximum must yield a scalar
    in [0.0, 1.0].
    """
    with torch.no_grad():
        logits = _FINGERSPELL_MODEL(x)           # (1, num_classes)
        probs = F.softmax(logits, dim=-1)         # (1, num_classes)
        confidence = probs.max().item()           # scalar

    assert isinstance(confidence, float), (
        f"Expected a float, got {type(confidence)}"
    )
    assert 0.0 <= confidence <= 1.0, (
        f"FingerspellingMLP confidence {confidence:.6f} is outside [0.0, 1.0] "
        f"for input shape {x.shape}"
    )


# ---------------------------------------------------------------------------
# Property 2b — FingerspellingMLP: full softmax distribution sums to ~1
# **Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability**
# **Validates: Requirements 2.2**
# ---------------------------------------------------------------------------

@given(x=fingerspell_input())
@settings(max_examples=100)
def test_fingerspell_mlp_softmax_sums_to_one(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 2.2**

    The full softmax probability distribution over all classes must sum to 1.0
    (within floating-point tolerance), confirming a valid probability simplex.
    """
    with torch.no_grad():
        logits = _FINGERSPELL_MODEL(x)
        probs = F.softmax(logits, dim=-1)
        prob_sum = probs.sum().item()

    assert abs(prob_sum - 1.0) < 1e-5, (
        f"FingerspellingMLP softmax probabilities sum to {prob_sum:.8f}, expected ~1.0"
    )


# ---------------------------------------------------------------------------
# Property 2c — SignLSTM confidence in [0.0, 1.0]
# **Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability**
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

@given(x=word_level_input())
@settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
def test_sign_lstm_confidence_is_valid_probability(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 3.2**

    For any float32 input tensor of shape [1, 30, 252], applying softmax to
    SignLSTM's output logits and taking the maximum must yield a scalar in
    [0.0, 1.0].
    """
    with torch.no_grad():
        logits = _SIGN_LSTM_MODEL(x)             # (1, num_classes)
        probs = F.softmax(logits, dim=-1)         # (1, num_classes)
        confidence = probs.max().item()           # scalar

    assert isinstance(confidence, float), (
        f"Expected a float, got {type(confidence)}"
    )
    assert 0.0 <= confidence <= 1.0, (
        f"SignLSTM confidence {confidence:.6f} is outside [0.0, 1.0] "
        f"for input shape {x.shape}"
    )


# ---------------------------------------------------------------------------
# Property 2d — SignLSTM: full softmax distribution sums to ~1
# **Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability**
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

@given(x=word_level_input())
@settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
def test_sign_lstm_softmax_sums_to_one(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 3.2**

    The full softmax probability distribution over all SignLSTM classes must sum
    to 1.0 (within floating-point tolerance).
    """
    with torch.no_grad():
        logits = _SIGN_LSTM_MODEL(x)
        probs = F.softmax(logits, dim=-1)
        prob_sum = probs.sum().item()

    assert abs(prob_sum - 1.0) < 1e-5, (
        f"SignLSTM softmax probabilities sum to {prob_sum:.8f}, expected ~1.0"
    )


# ---------------------------------------------------------------------------
# Property 2e — individual probabilities are non-negative
# **Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability**
# **Validates: Requirements 2.2, 3.2**
# ---------------------------------------------------------------------------

@given(x=fingerspell_input())
@settings(max_examples=100)
def test_fingerspell_mlp_all_probs_non_negative(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 2.2**

    Every individual probability in the FingerspellingMLP softmax output must
    be ≥ 0.0 (since softmax never produces negative values from real logits).
    """
    with torch.no_grad():
        logits = _FINGERSPELL_MODEL(x)
        probs = F.softmax(logits, dim=-1)
        min_prob = probs.min().item()

    assert min_prob >= 0.0, (
        f"FingerspellingMLP produced a negative probability: {min_prob:.8f}"
    )


@given(x=word_level_input())
@settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
def test_sign_lstm_all_probs_non_negative(x: torch.Tensor) -> None:
    """
    # Feature: sign-language-interpreter, Property 2: Classifier confidence is always a valid probability
    **Validates: Requirements 3.2**

    Every individual probability in the SignLSTM softmax output must be ≥ 0.0.
    """
    with torch.no_grad():
        logits = _SIGN_LSTM_MODEL(x)
        probs = F.softmax(logits, dim=-1)
        min_prob = probs.min().item()

    assert min_prob >= 0.0, (
        f"SignLSTM produced a negative probability: {min_prob:.8f}"
    )
