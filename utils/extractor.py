"""
ConfidenceGate — pre-classification filter for MediaPipe detection results.

Suppresses classifier inference on frames where MediaPipe detection confidence
is below threshold or too many landmarks have low visibility.  This is the
canonical Python implementation used by the server-side pipeline.

Requirements: 1.7
"""

from __future__ import annotations

# Thresholds defined in design.md (Property 1 / Requirement 1.7)
_DETECTION_CONF_THRESHOLD: float = 0.8
_VISIBLE_LM_THRESHOLD: int = 18


class ConfidenceGate:
    """Pre-classification filter based on MediaPipe per-hand detection quality.

    A frame is allowed through the gate only when BOTH of the following
    conditions are satisfied:

    - ``detection_conf >= 0.8``  — MediaPipe's per-hand detection confidence
    - ``visible_lm_count >= 18`` — number of landmarks with visibility ≥ 0.5

    Frames that fail either condition are silently dropped; the caller should
    hold the last valid prediction in the output display (Requirement 1.4).

    Args:
        conf_threshold:       Minimum detection confidence (default 0.8).
        min_visible_landmarks: Minimum number of visible landmarks (default 18).
    """

    def __init__(
        self,
        conf_threshold: float = _DETECTION_CONF_THRESHOLD,
        min_visible_landmarks: int = _VISIBLE_LM_THRESHOLD,
    ) -> None:
        if not (0.0 <= conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in [0, 1]")
        if min_visible_landmarks < 0:
            raise ValueError("min_visible_landmarks must be >= 0")

        self.conf_threshold = conf_threshold
        self.min_visible_landmarks = min_visible_landmarks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_classify(
        self,
        detection_conf: float,
        visible_lm_count: int,
    ) -> bool:
        """Decide whether this frame is of sufficient quality to run inference.

        Args:
            detection_conf:   MediaPipe per-hand detection confidence in [0, 1].
            visible_lm_count: Number of landmarks whose MediaPipe visibility
                              score is >= 0.5.

        Returns:
            ``True`` if BOTH thresholds are met and the frame should be
            forwarded to the classifier; ``False`` otherwise.
        """
        return (
            detection_conf >= self.conf_threshold
            and visible_lm_count >= self.min_visible_landmarks
        )
