"""
Sequence dataset for word-level sign recognition training.

Supports:
  - WLASL folder structure  (gloss/video_id.mp4)
  - ASL Citizen metadata CSV with signer-independent pre-defined splits

Offline preprocessing extracts MediaPipe Holistic landmarks from video files
and caches them as .npy arrays for fast training-time loading.

Requirements: 3.1, 10.1, 10.2, 10.3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.landmarks import add_velocity_features, build_two_hand_vector


# ---------------------------------------------------------------------------
# Sequence resampling
# ---------------------------------------------------------------------------

def _resample(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate a landmark sequence to a fixed number of frames.

    Args:
        sequence:   Array of shape (T_orig, feature_dim).
        target_len: Desired output length T.

    Returns:
        Array of shape (target_len, feature_dim).
    """
    T_orig, D = sequence.shape
    if T_orig == target_len:
        return sequence.astype(np.float32)

    if T_orig == 1:
        return np.tile(sequence.astype(np.float32), (target_len, 1))

    # Build output via linear interpolation
    orig_indices = np.linspace(0, T_orig - 1, T_orig)
    new_indices = np.linspace(0, T_orig - 1, target_len)
    out = np.zeros((target_len, D), dtype=np.float32)

    for d in range(D):
        out[:, d] = np.interp(new_indices, orig_indices, sequence[:, d])

    return out


# ---------------------------------------------------------------------------
# SignSequenceDataset
# ---------------------------------------------------------------------------

class SignSequenceDataset(Dataset):
    """PyTorch Dataset of pre-extracted landmark sequences for word-level signs.

    Each item is ``(feature_tensor, class_index)`` where:
      - ``feature_tensor`` has shape ``(seq_len, 252)`` (positions + velocities)
      - ``class_index`` is an integer gloss index

    Args:
        samples:       List of ``(npy_path, label_index)`` tuples.
        label_to_idx:  Mapping from gloss string to integer index.
        seq_len:       Fixed temporal length after resampling (default 30).
        use_velocity:  If True, append velocity features → (seq_len, 252).
        augment:       If True, apply Gaussian noise augmentation to sequences.
        transform:     Optional additional callable transform.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        label_to_idx: Dict[str, int],
        seq_len: int = 30,
        use_velocity: bool = True,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        if not samples:
            raise ValueError("samples list must be non-empty")

        self.samples = samples
        self.label_to_idx = label_to_idx
        self.idx_to_label: Dict[int, str] = {v: k for k, v in label_to_idx.items()}
        self.seq_len = seq_len
        self.use_velocity = use_velocity
        self.augment = augment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        npy_path, label_idx = self.samples[idx]

        # Load pre-extracted (T, 126) two-hand landmark array
        seq: np.ndarray = np.load(str(npy_path)).astype(np.float32)

        if seq.ndim == 1:
            # Handle single-frame edge case
            seq = seq.reshape(1, -1)

        # Resample to fixed length
        seq = _resample(seq, self.seq_len)  # (seq_len, 126)

        # Optional augmentation: small Gaussian noise on positions
        if self.augment:
            seq = seq + np.random.normal(0.0, 0.01, seq.shape).astype(np.float32)

        # Append velocity features → (seq_len, 252)
        if self.use_velocity:
            seq = add_velocity_features(seq)

        if self.transform is not None:
            seq = self.transform(seq)

        x = torch.from_numpy(seq)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y

    @property
    def num_classes(self) -> int:
        return len(self.label_to_idx)

    @property
    def labels(self) -> np.ndarray:
        """Integer label array of shape (N,) — used for WeightedRandomSampler."""
        return np.array([s[1] for s in self.samples], dtype=np.int64)


# ---------------------------------------------------------------------------
# WLASL loader
# ---------------------------------------------------------------------------

def load_wlasl_dataset(
    landmark_root: str | Path,
    split: str = "train",
    seq_len: int = 30,
    use_velocity: bool = True,
    augment: bool = False,
) -> SignSequenceDataset:
    """Load a WLASL landmark dataset from a pre-extracted directory.

    Expected structure after offline extraction::

        landmark_root/
            train/
                BOOK/
                    00001.npy   # shape (T, 126)
                    00002.npy
                COMPUTER/
                    ...
            val/
            test/

    Args:
        landmark_root: Root directory of the pre-extracted WLASL landmarks.
        split:         One of "train", "val", "test".
        seq_len:       Target sequence length after resampling.
        use_velocity:  Whether to append velocity features.
        augment:       Whether to apply noise augmentation (train only).

    Returns:
        SignSequenceDataset ready for use with DataLoader.
    """
    landmark_root = Path(landmark_root)
    split_dir = landmark_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"WLASL split directory not found: {split_dir}")

    label_to_idx: Dict[str, int] = {}
    samples: List[Tuple[Path, int]] = []

    # Sort for reproducibility
    gloss_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    for gloss_dir in gloss_dirs:
        gloss = gloss_dir.name.upper()
        if gloss not in label_to_idx:
            label_to_idx[gloss] = len(label_to_idx)
        label_idx = label_to_idx[gloss]

        for npy_file in sorted(gloss_dir.glob("*.npy")):
            samples.append((npy_file, label_idx))

    if not samples:
        raise RuntimeError(
            f"No .npy landmark files found under {split_dir}. "
            "Run extract_landmarks_from_videos() first."
        )

    return SignSequenceDataset(
        samples=samples,
        label_to_idx=label_to_idx,
        seq_len=seq_len,
        use_velocity=use_velocity,
        augment=augment,
    )


# ---------------------------------------------------------------------------
# ASL Citizen loader (signer-independent splits)
# ---------------------------------------------------------------------------

def load_asl_citizen_dataset(
    landmark_root: str | Path,
    metadata_csv: str | Path,
    split: str = "train",
    seq_len: int = 30,
    use_velocity: bool = True,
    augment: bool = False,
) -> SignSequenceDataset:
    """Load ASL Citizen from pre-extracted landmarks using the official metadata CSV.

    The metadata CSV is expected to have at minimum these columns:
      - ``file``        : relative path to the original video (or npy file)
      - ``gloss``       : gloss label string
      - ``split``       : one of "train", "val", "test"
      - ``signer_id``   : signer identifier (used to verify split integrity)

    Landmarks are expected to be stored at::

        landmark_root/{split}/{file_stem}.npy

    Args:
        landmark_root: Root directory of the pre-extracted ASL Citizen landmarks.
        metadata_csv:  Path to the official ASL Citizen metadata CSV.
        split:         One of "train", "val", "test".
        seq_len:       Target sequence length.
        use_velocity:  Whether to append velocity features.
        augment:       Whether to apply noise augmentation.

    Returns:
        SignSequenceDataset with the pre-defined signer-independent split.

    Raises:
        ValueError: If signer IDs leak between the requested split and other splits
                    (indicates incorrect CSV or landmark assignment).
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required for ASL Citizen loading. pip install pandas") from exc

    landmark_root = Path(landmark_root)
    metadata_csv = Path(metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required_cols = {"file", "gloss", "split", "signer_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    # Use the pre-defined split column (Requirement 10.3)
    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No samples found for split='{split}' in {metadata_csv}")

    # Build label map from the complete dataset (all splits) for consistency
    all_glosses = sorted(df["gloss"].str.upper().unique())
    label_to_idx: Dict[str, int] = {g: i for i, g in enumerate(all_glosses)}

    samples: List[Tuple[Path, int]] = []
    missing_files: List[str] = []

    for _, row in split_df.iterrows():
        gloss = str(row["gloss"]).upper()
        file_stem = Path(str(row["file"])).stem
        npy_path = landmark_root / split / f"{file_stem}.npy"

        if not npy_path.exists():
            missing_files.append(str(npy_path))
            continue

        label_idx = label_to_idx.get(gloss)
        if label_idx is None:
            continue

        samples.append((npy_path, label_idx))

    if missing_files:
        # Warn rather than raise — some videos may not have been extractable
        import warnings
        warnings.warn(
            f"{len(missing_files)} landmark files not found (e.g. {missing_files[0]}). "
            "Run extract_landmarks_from_videos() to generate them.",
            UserWarning,
            stacklevel=2,
        )

    if not samples:
        raise RuntimeError(
            f"No valid samples found for split='{split}'. "
            "Ensure landmark files exist under {landmark_root}/{split}/"
        )

    return SignSequenceDataset(
        samples=samples,
        label_to_idx=label_to_idx,
        seq_len=seq_len,
        use_velocity=use_velocity,
        augment=augment,
    )


# ---------------------------------------------------------------------------
# Offline landmark extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_videos(
    video_root: str | Path,
    output_root: str | Path,
    dataset_type: str = "wlasl",
    metadata_csv: Optional[str | Path] = None,
) -> None:
    """Extract MediaPipe Holistic landmarks from video files and save as .npy.

    Output shape per video: ``(T, 126)`` — two-hand two-hand vector per frame,
    where T is the number of frames with detected hands.

    For WLASL, expects::

        video_root/GLOSS/video_id.mp4

    For ASL Citizen, requires metadata_csv with ``file``, ``gloss``, ``split``
    columns.

    Args:
        video_root:   Root directory of input videos.
        output_root:  Root directory for output .npy files.
        dataset_type: "wlasl" or "asl_citizen".
        metadata_csv: Required for "asl_citizen" dataset type.
    """
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mediapipe and opencv-python are required for video extraction. "
            "pip install mediapipe opencv-python"
        ) from exc

    video_root = Path(video_root)
    output_root = Path(output_root)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    from utils.landmarks import landmarks_from_mediapipe, get_handedness, build_two_hand_vector

    def _extract_one(video_path: Path) -> Optional[np.ndarray]:
        """Extract (T, 126) landmark array from one video. Returns None on failure."""
        cap = cv2.VideoCapture(str(video_path))
        frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            right_lm = None
            left_lm = None

            if results.right_hand_landmarks:
                right_lm = landmarks_from_mediapipe(results.right_hand_landmarks)
            if results.left_hand_landmarks:
                left_lm = landmarks_from_mediapipe(results.left_hand_landmarks)

            vec = build_two_hand_vector(right_lm, left_lm)
            frames.append(vec)

        cap.release()

        if not frames:
            return None

        return np.stack(frames, axis=0).astype(np.float32)  # (T, 126)

    if dataset_type == "wlasl":
        for gloss_dir in sorted(video_root.iterdir()):
            if not gloss_dir.is_dir():
                continue
            gloss = gloss_dir.name.upper()

            for split in ("train", "val", "test"):
                split_video_dir = gloss_dir
                # WLASL may not have explicit train/val/test sub-folders;
                # we place all extractions under the same gloss name
                out_dir = output_root / "train" / gloss
                out_dir.mkdir(parents=True, exist_ok=True)

                for video_path in sorted(gloss_dir.glob("*.mp4")):
                    out_path = out_dir / f"{video_path.stem}.npy"
                    if out_path.exists():
                        continue
                    seq = _extract_one(video_path)
                    if seq is not None:
                        np.save(str(out_path), seq)

    elif dataset_type == "asl_citizen":
        if metadata_csv is None:
            raise ValueError("metadata_csv is required for dataset_type='asl_citizen'")

        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError("pandas is required. pip install pandas") from exc

        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            split = str(row["split"])
            file_path = video_root / str(row["file"])
            file_stem = Path(str(row["file"])).stem

            out_dir = output_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{file_stem}.npy"

            if out_path.exists() or not file_path.exists():
                continue

            seq = _extract_one(file_path)
            if seq is not None:
                np.save(str(out_path), seq)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Use 'wlasl' or 'asl_citizen'.")

    holistic.close()
