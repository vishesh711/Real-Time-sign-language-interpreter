"""
PredictionGate — three-layer temporal filter applied after every inference call.

Layer 1: Majority vote over a sliding window of the last N frames.
Layer 2: Hold queue — the voted label must be stable for M consecutive frames.
Layer 3: Cooldown — a minimum gap of K frames between two accepted predictions.

This is the canonical Python implementation; the browser TypeScript version in
``browser/gate.ts`` must mirror this logic exactly.

Requirements: 2.4, 9.5, 9.6
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class PredictionGate:
    """Three-layer temporal filter for stable sign commitment.

    Args:
        vote_window:    Number of recent frames used for majority voting (default 7).
        hold_frames:    Number of consecutive frames the voted label must be
                        stable before it is accepted (default 12).
        confidence_threshold: Minimum per-frame confidence required for a frame
                        to contribute to the vote window (default 0.8).
        cooldown_frames: Number of frames to suppress new predictions after an
                        acceptance event (default 20).
    """

    def __init__(
        self,
        vote_window: int = 7,
        hold_frames: int = 12,
        confidence_threshold: float = 0.8,
        cooldown_frames: int = 20,
    ) -> None:
        if vote_window < 1:
            raise ValueError("vote_window must be >= 1")
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be >= 0")

        self.vote_window = vote_window
        self.hold_frames = hold_frames
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames

        # Internal state
        self._window: deque[str] = deque(maxlen=vote_window)
        self._hold_label: Optional[str] = None
        self._hold_count: int = 0
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, label: str, confidence: float) -> Optional[str]:
        """Feed one frame's prediction into the gate.

        Args:
            label:      Predicted class label for this frame.
            confidence: Softmax probability in [0.0, 1.0] for the predicted label.

        Returns:
            The accepted label string if the gate fires, or ``None`` if no
            commitment is made this frame.
        """
        # Decrement cooldown counter
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            # Still suppress: push to window but do not accept
            if confidence >= self.confidence_threshold:
                self._window.append(label)
            else:
                self._window.append("")  # low-confidence frame counts as no vote
            # Reset hold state during cooldown
            self._hold_label = None
            self._hold_count = 0
            return None

        # Layer 1 — majority vote
        if confidence >= self.confidence_threshold:
            self._window.append(label)
        else:
            self._window.append("")  # low-confidence frame: blank vote

        voted_label = self._majority_vote()

        if voted_label is None:
            # No majority — reset hold
            self._hold_label = None
            self._hold_count = 0
            return None

        # Layer 2 — hold queue
        if voted_label == self._hold_label:
            self._hold_count += 1
        else:
            self._hold_label = voted_label
            self._hold_count = 1

        if self._hold_count >= self.hold_frames:
            # Layer 3 — commit and enter cooldown
            accepted = self._hold_label
            self._hold_label = None
            self._hold_count = 0
            self._window.clear()
            self._cooldown_remaining = self.cooldown_frames
            return accepted

        return None

    def reset(self) -> None:
        """Clear all internal state (call when the user clears the buffer or
        the hand is lost)."""
        self._window.clear()
        self._hold_label = None
        self._hold_count = 0
        self._cooldown_remaining = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _majority_vote(self) -> Optional[str]:
        """Return the label with a strict majority (> 50 %) in the window,
        or None if no such label exists.  Blank votes ("") never win."""
        if not self._window:
            return None

        counts: dict[str, int] = {}
        for lbl in self._window:
            if lbl:  # ignore blank votes
                counts[lbl] = counts.get(lbl, 0) + 1

        if not counts:
            return None

        best_label = max(counts, key=lambda k: counts[k])
        if counts[best_label] > len(self._window) / 2:
            return best_label
        return None
