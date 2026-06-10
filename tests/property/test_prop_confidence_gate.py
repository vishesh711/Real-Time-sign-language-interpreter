"""
Property-based test for ConfidenceGate.

# Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames
**Validates: Requirements 1.7**

Property 1 — Confidence gate rejects low-quality frames:
    For any MediaPipe result where detection_confidence < 0.8 OR visible_landmarks < 18,
    the confidence gate SHALL prevent classifier inference from running.
    Conversely, for any result where both thresholds are met, the gate SHALL
    pass the landmark array to the classifier.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.extractor import ConfidenceGate

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Detection confidence: any float in [0.0, 1.0]
detection_conf_st = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Visible landmark count: any integer in [0, 21]
visible_lm_count_st = st.integers(min_value=0, max_value=21)


# ---------------------------------------------------------------------------
# Property 1a — gate rejects frames below EITHER threshold
# **Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames**
# **Validates: Requirements 1.7**
# ---------------------------------------------------------------------------

@given(
    conf=detection_conf_st.filter(lambda c: c < 0.8),
    count=visible_lm_count_st,
)
@settings(max_examples=100)
def test_confidence_gate_rejects_low_confidence(conf: float, count: int) -> None:
    """
    # Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames
    **Validates: Requirements 1.7**

    When detection_confidence < 0.8 (regardless of visible landmark count),
    ConfidenceGate.should_classify() SHALL return False.
    """
    gate = ConfidenceGate()
    result = gate.should_classify(conf, count)
    assert result is False, (
        f"Expected should_classify({conf}, {count}) == False "
        f"(conf={conf} < 0.8 triggers rejection), but got True"
    )


@given(
    conf=detection_conf_st,
    count=visible_lm_count_st.filter(lambda c: c < 18),
)
@settings(max_examples=100)
def test_confidence_gate_rejects_low_landmark_count(conf: float, count: int) -> None:
    """
    # Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames
    **Validates: Requirements 1.7**

    When visible_landmark_count < 18 (regardless of detection confidence),
    ConfidenceGate.should_classify() SHALL return False.
    """
    gate = ConfidenceGate()
    result = gate.should_classify(conf, count)
    assert result is False, (
        f"Expected should_classify({conf}, {count}) == False "
        f"(count={count} < 18 triggers rejection), but got True"
    )


@given(
    conf=detection_conf_st.filter(lambda c: c < 0.8),
    count=visible_lm_count_st.filter(lambda c: c < 18),
)
@settings(max_examples=100)
def test_confidence_gate_rejects_when_both_thresholds_fail(conf: float, count: int) -> None:
    """
    # Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames
    **Validates: Requirements 1.7**

    When BOTH detection_confidence < 0.8 AND visible_landmark_count < 18,
    ConfidenceGate.should_classify() SHALL return False.
    """
    gate = ConfidenceGate()
    result = gate.should_classify(conf, count)
    assert result is False, (
        f"Expected should_classify({conf}, {count}) == False "
        f"(both thresholds fail), but got True"
    )


# ---------------------------------------------------------------------------
# Property 1b — gate passes frames meeting BOTH thresholds
# **Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames**
# **Validates: Requirements 1.7**
# ---------------------------------------------------------------------------

@given(
    conf=detection_conf_st.filter(lambda c: c >= 0.8),
    count=visible_lm_count_st.filter(lambda c: c >= 18),
)
@settings(max_examples=100)
def test_confidence_gate_passes_high_quality_frames(conf: float, count: int) -> None:
    """
    # Feature: sign-language-interpreter, Property 1: Confidence gate rejects low-quality frames
    **Validates: Requirements 1.7**

    When detection_confidence >= 0.8 AND visible_landmark_count >= 18,
    ConfidenceGate.should_classify() SHALL return True — allowing inference to run.
    """
    gate = ConfidenceGate()
    result = gate.should_classify(conf, count)
    assert result is True, (
        f"Expected should_classify({conf}, {count}) == True "
        f"(both thresholds met), but got False"
    )
