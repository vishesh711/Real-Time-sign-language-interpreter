"""
Landmark normalization utilities for the Sign Language Interpreter.

Converts raw MediaPipe output into normalized feature vectors suitable
for the FingerspellMLP (63-float) and SignLSTM (126-float → 252-float) models.

Normalization strategy:
  1. Translate so the wrist (landmark 0) is at the origin.
  2. Scale by the distance from wrist to index-finger MCP (landmark 5).
  3. For left-hand landmarks, mirror the x-axis so the classifier sees a
     canonical "right-hand" representation.

Requirements: 1.3, 1.7, 3.1
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Label maps (A–Z fingerspelling + digits 0–9)
# ---------------------------------------------------------------------------

_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIGITS = [str(d) for d in range(10)]
_ALL_LABELS = _LETTERS + _DIGITS

LABEL_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(_ALL_LABELS)}
IDX_TO_LABEL: dict[int, str] = {idx: label for label, idx in LABEL_TO_IDX.items()}

# MediaPipe hand landmark indices used for normalization
_WRIST_IDX = 0
_INDEX_MCP_IDX = 5  # index-finger metacarpo-phalangeal joint


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def normalize_landmarks(
    hand_lm: np.ndarray,
    mirror_left: bool = False,
) -> np.ndarray | None:
    """Normalize a single hand's 21 landmarks to a canonical 63-float vector.

    Args:
        hand_lm:    Array of shape (21, 3) — raw MediaPipe hand landmarks.
        mirror_left: When True, flip the x-axis so left-hand poses align with
                     right-hand poses in the classifier's feature space.

    Returns:
        Flattened array of shape (63,), or None if the input is degenerate
        (e.g. all-zero wrist, zero scale distance).
    """
    if hand_lm is None or hand_lm.shape != (21, 3):
        return None

    lm = hand_lm.astype(np.float32).copy()

    # 1. Translate to wrist origin
    wrist = lm[_WRIST_IDX].copy()
    lm -= wrist

    # 2. Scale by wrist → index-MCP distance
    scale = float(np.linalg.norm(lm[_INDEX_MCP_IDX]))
    if scale < 1e-6:
        return None
    lm /= scale

    # 3. Mirror left-hand x-axis for canonical orientation
    if mirror_left:
        lm[:, 0] *= -1.0

    return lm.flatten()  # (63,)


def landmarks_from_mediapipe(
    mp_hand_landmarks,
    image_width: int = 1,
    image_height: int = 1,
) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmarkList to a (21, 3) numpy array.

    MediaPipe normalized landmarks are in [0, 1] for x/y and represent depth
    for z.  We preserve them as-is; normalization happens in normalize_landmarks().

    Args:
        mp_hand_landmarks: MediaPipe hand landmark object with .landmark list.
        image_width:  Frame width (unused — kept for interface compatibility).
        image_height: Frame height (unused — kept for interface compatibility).

    Returns:
        Array of shape (21, 3) with (x, y, z) per keypoint.
    """
    lm_list = []
    for lm in mp_hand_landmarks.landmark:
        lm_list.append([lm.x, lm.y, lm.z])
    return np.array(lm_list, dtype=np.float32)


def get_handedness(mp_handedness) -> str:
    """Extract the dominant label ('Left' or 'Right') from a MediaPipe
    ClassificationList for a detected hand.

    Args:
        mp_handedness: MediaPipe handedness classification object.

    Returns:
        'Left' or 'Right'.
    """
    return mp_handedness.classification[0].label


# ---------------------------------------------------------------------------
# Two-hand vector builder
# ---------------------------------------------------------------------------

def build_two_hand_vector(
    right_lm: np.ndarray | None,
    left_lm: np.ndarray | None,
) -> np.ndarray:
    """Combine right and left hand normalized landmarks into a 126-float vector.

    Missing hands are represented as all-zeros.  The left hand is mirrored
    before concatenation so both hands share the same coordinate convention.

    Args:
        right_lm: Normalized right-hand landmarks of shape (21, 3), or None.
        left_lm:  Normalized left-hand landmarks of shape (21, 3), or None.

    Returns:
        Array of shape (126,): [right_63 | left_63].
    """
    zeros = np.zeros(63, dtype=np.float32)

    if right_lm is not None:
        right_vec = normalize_landmarks(right_lm, mirror_left=False)
        right_part = right_vec if right_vec is not None else zeros.copy()
    else:
        right_part = zeros.copy()

    if left_lm is not None:
        left_vec = normalize_landmarks(left_lm, mirror_left=True)
        left_part = left_vec if left_vec is not None else zeros.copy()
    else:
        left_part = zeros.copy()

    return np.concatenate([right_part, left_part], axis=0)  # (126,)


# ---------------------------------------------------------------------------
# Velocity feature builder  (Requirements: 3.1)
# ---------------------------------------------------------------------------

def add_velocity_features(sequence: np.ndarray) -> np.ndarray:
    """Append frame-over-frame delta features to a landmark sequence.

    For each frame t the velocity is defined as ``sequence[t] - sequence[t-1]``.
    Frame 0 receives a zero-padded velocity (no prior frame).

    Args:
        sequence: Array of shape (T, 126) — two-hand landmark vectors.

    Returns:
        Array of shape (T, 252) where columns 0:126 are the original positions
        and columns 126:252 are the per-frame velocity deltas.

    Raises:
        ValueError: If the input does not have shape (T, 126).
    """
    if sequence.ndim != 2 or sequence.shape[1] != 126:
        raise ValueError(
            f"Expected input shape (T, 126), got {sequence.shape}"
        )

    T = sequence.shape[0]
    velocity = np.zeros_like(sequence, dtype=np.float32)

    # velocity[0] stays zero (no previous frame)
    if T > 1:
        velocity[1:] = sequence[1:].astype(np.float32) - sequence[:-1].astype(np.float32)

    positions = sequence.astype(np.float32)
    return np.concatenate([positions, velocity], axis=1)  # (T, 252)
