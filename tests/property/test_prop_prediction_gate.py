"""
Property-based test for PredictionGate.

**Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run**
**Validates: Requirements 2.4, 9.5, 9.6**

Property 3 — PredictionGate commits exactly once per stable run:
    For any sequence of per-frame predictions fed into PredictionGate, if a
    single label occupies the majority of the vote window AND appears in every
    slot of the hold queue AND exceeds the confidence threshold AND the cooldown
    has expired, then the gate SHALL fire exactly once — emitting the label —
    and immediately enter cooldown, preventing a second emission from the same run.
"""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from utils.gate import PredictionGate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = list(string.ascii_uppercase[:26])  # A–Z

# A "stable run" needs vote_window + hold_frames frames to guarantee a commit.
# For the "exactly once" property to hold, the cooldown must be long enough
# that the gate cannot re-trigger a second commit within those same frames.
# Minimum safe cooldown: vote_window + hold_frames frames so the gate is still
# suppressed when the last frame of the run is processed.
_MAX_WINDOW = 10
_MAX_HOLD = 10


@st.composite
def gate_params_with_safe_cooldown(draw):
    """
    Draw gate params where cooldown_frames >= vote_window + hold_frames.
    This ensures that a stable run of (vote_window + hold_frames) frames
    can produce at most one commit before the cooldown blocks any second commit.
    """
    vote_window = draw(st.integers(min_value=1, max_value=_MAX_WINDOW))
    hold_frames = draw(st.integers(min_value=1, max_value=_MAX_HOLD))
    confidence_threshold = draw(st.floats(min_value=0.0, max_value=0.95, allow_nan=False))
    stable_run_length = vote_window + hold_frames
    # cooldown must be at least as long as the stable run so re-triggering
    # is impossible within a single run of stable_run_length frames
    cooldown_frames = draw(st.integers(min_value=stable_run_length, max_value=stable_run_length + 20))
    return {
        "vote_window": vote_window,
        "hold_frames": hold_frames,
        "confidence_threshold": confidence_threshold,
        "cooldown_frames": cooldown_frames,
    }


@st.composite
def gate_params_any(draw):
    """Draw any valid PredictionGate constructor parameters."""
    vote_window = draw(st.integers(min_value=1, max_value=_MAX_WINDOW))
    hold_frames = draw(st.integers(min_value=1, max_value=_MAX_HOLD))
    confidence_threshold = draw(st.floats(min_value=0.0, max_value=0.95, allow_nan=False))
    cooldown_frames = draw(st.integers(min_value=0, max_value=30))
    return {
        "vote_window": vote_window,
        "hold_frames": hold_frames,
        "confidence_threshold": confidence_threshold,
        "cooldown_frames": cooldown_frames,
    }


# ---------------------------------------------------------------------------
# Property 3 — commits exactly once per stable run
# **Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run**
# **Validates: Requirements 2.4, 9.5, 9.6**
# ---------------------------------------------------------------------------

@given(params=gate_params_with_safe_cooldown(), label=st.sampled_from(LABELS))
@settings(max_examples=200)
def test_gate_commits_exactly_once_per_stable_run(params, label):
    """
    **Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run**
    **Validates: Requirements 2.4, 9.5, 9.6**

    For any gate configuration (with cooldown long enough to cover the stable run),
    feeding vote_window + hold_frames identical high-confidence predictions must
    produce exactly one commitment — not zero and not more than one.
    """
    gate = PredictionGate(**params)
    confidence = min(params["confidence_threshold"] + 0.05, 1.0)

    stable_run_length = gate.vote_window + gate.hold_frames
    accepted: list[str] = []
    for _ in range(stable_run_length):
        result = gate.update(label, confidence)
        if result is not None:
            accepted.append(result)

    assert len(accepted) == 1, (
        f"Expected exactly 1 commitment, got {len(accepted)} "
        f"(params={params}, label={label})"
    )
    assert accepted[0] == label, (
        f"Committed wrong label: expected {label!r}, got {accepted[0]!r}"
    )


@given(params=gate_params_with_safe_cooldown(), label=st.sampled_from(LABELS))
@settings(max_examples=200)
def test_gate_does_not_recommit_during_cooldown(params, label):
    """
    **Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run**
    **Validates: Requirements 2.4, 9.5, 9.6**

    Immediately after a commit, the gate enters cooldown. Feeding the same
    high-confidence label for exactly cooldown_frames frames must NOT produce
    a second commit.
    """
    gate = PredictionGate(**params)
    confidence = min(params["confidence_threshold"] + 0.05, 1.0)

    # Drive to the first commit.
    # Track how many frames have been consumed after the commit so we know
    # how many cooldown ticks have already been used by the stable-run loop.
    commit_frame_idx: int | None = None
    stable_run_length = gate.vote_window + gate.hold_frames
    for i in range(stable_run_length):
        result = gate.update(label, confidence)
        if result is not None and commit_frame_idx is None:
            commit_frame_idx = i

    if commit_frame_idx is None:
        # Gate never fired — skip (shouldn't happen with safe_cooldown params)
        return

    # Frames consumed after the commit within the stable-run loop already
    # decremented the cooldown counter that many times.
    frames_after_commit_in_loop = (stable_run_length - 1) - commit_frame_idx
    remaining_cooldown = gate.cooldown_frames - frames_after_commit_in_loop

    if remaining_cooldown <= 0:
        # All cooldown already drained by the stable-run loop; nothing to test.
        return

    # Feed for exactly `remaining_cooldown` more frames — all must be suppressed.
    # In gate.update(), when _cooldown_remaining > 0 it decrements and returns None,
    # so the frame that brings cooldown from 1 → 0 also returns None.
    during_cooldown: list[str] = []
    for _ in range(remaining_cooldown):
        result = gate.update(label, confidence)
        if result is not None:
            during_cooldown.append(result)

    assert len(during_cooldown) == 0, (
        f"Gate committed {len(during_cooldown)} extra time(s) during cooldown "
        f"(params={params}, label={label}, "
        f"commit_frame_idx={commit_frame_idx}, remaining_cooldown={remaining_cooldown})"
    )


@given(params=gate_params_any(), label=st.sampled_from(LABELS))
@settings(max_examples=200)
def test_gate_emits_correct_label(params, label):
    """
    **Feature: sign-language-interpreter, Property 3: PredictionGate commits exactly once per stable run**
    **Validates: Requirements 2.4, 9.5, 9.6**

    Whenever the gate commits, it must emit exactly the label that held majority
    in the vote window. Running a long all-identical sequence and collecting all
    emissions: every emission must equal the input label.
    """
    gate = PredictionGate(**params)
    confidence = min(params["confidence_threshold"] + 0.05, 1.0)

    # Run for many frames; collect all commits
    total_frames = (gate.vote_window + gate.hold_frames + gate.cooldown_frames + 1) * 3
    for _ in range(total_frames):
        result = gate.update(label, confidence)
        if result is not None:
            assert result == label, (
                f"Gate emitted {result!r} but input label was {label!r} "
                f"(params={params})"
            )
