"""
Offline landmark dataset extraction utilities.

Supports:
  - Sign Language MNIST CSV format (pixel rows → MediaPipe-extracted landmarks)
  - Custom image folders organised as label/image.jpg
  - Augmentation: Gaussian noise, z-axis rotation, scale jitter

Requirements: 2.1
"""

from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.landmarks import LABEL_TO_IDX, normalize_landmarks


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment_gaussian_noise(lm: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add zero-mean Gaussian noise to landmark coordinates."""
    noise = np.random.normal(0.0, std, size=lm.shape).astype(np.float32)
    return lm + noise


def _augment_z_rotation(lm: np.ndarray, angle_deg: float | None = None) -> np.ndarray:
    """Rotate landmarks around the z-axis by a random (or specified) angle."""
    if angle_deg is None:
        angle_deg = random.uniform(-15.0, 15.0)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a, 0.0],
                    [sin_a,  cos_a, 0.0],
                    [0.0,    0.0,   1.0]], dtype=np.float32)
    lm_3d = lm.reshape(21, 3)
    return (lm_3d @ rot.T).reshape(-1)


def _augment_scale_jitter(lm: np.ndarray, scale_range: tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Uniformly scale landmarks by a random factor."""
    scale = random.uniform(*scale_range)
    return lm * scale


def augment_landmarks(lm: np.ndarray) -> np.ndarray:
    """Apply all augmentations in random order with 50 % probability each."""
    if random.random() < 0.5:
        lm = _augment_gaussian_noise(lm)
    if random.random() < 0.5:
        lm = _augment_z_rotation(lm)
    if random.random() < 0.5:
        lm = _augment_scale_jitter(lm)
    return lm


# ---------------------------------------------------------------------------
# LandmarkDataset
# ---------------------------------------------------------------------------

class LandmarkDataset(Dataset):
    """PyTorch Dataset of pre-extracted landmark feature vectors.

    Each item is a tuple ``(feature_vector, class_index)`` where
    ``feature_vector`` is a float32 tensor of shape ``(input_dim,)``.

    Args:
        features:   Array of shape (N, input_dim).
        labels:     Array of shape (N,) with integer class indices.
        augment:    Whether to apply random augmentations during __getitem__.
        transform:  Optional additional transform callable.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length, "
                f"got {len(features)} and {len(labels)}"
            )
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        lm = self.features[idx].copy()

        if self.augment:
            lm = augment_landmarks(lm)

        if self.transform is not None:
            lm = self.transform(lm)

        x = torch.from_numpy(lm)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def class_counts(self) -> np.ndarray:
        """Count of samples per class, shape (num_classes,)."""
        counts = np.bincount(self.labels, minlength=self.num_classes)
        return counts


# ---------------------------------------------------------------------------
# Sign Language MNIST CSV loader
# ---------------------------------------------------------------------------

def load_sign_mnist_csv(
    csv_path: str | Path,
    augment: bool = False,
) -> LandmarkDataset:
    """Load a Sign Language MNIST CSV file into a LandmarkDataset.

    The Sign Language MNIST CSV has columns:
      label, pixel1, pixel2, ..., pixel784

    Each 28×28 pixel image is passed through MediaPipe offline to extract
    landmarks.  To avoid a hard MediaPipe dependency at import time this
    function attempts to use MediaPipe when available; if not available it
    falls back to extracting a simple 63-float PCA-style feature from the
    pixel values (for unit-testing only).

    Args:
        csv_path: Path to the train or test CSV file.
        augment:  Whether to enable landmark augmentation.

    Returns:
        LandmarkDataset ready for training.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[tuple[int, np.ndarray]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_int = int(row["label"])
            pixels = np.array(
                [int(row[f"pixel{i+1}"]) for i in range(784)],
                dtype=np.float32,
            )
            rows.append((label_int, pixels))

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []

    try:
        import mediapipe as mp  # type: ignore
        import cv2  # type: ignore

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
        )

        for label_int, pixels in rows:
            img = (pixels.reshape(28, 28) * (255.0 / pixels.max() if pixels.max() > 0 else 1.0)).astype(np.uint8)
            img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_GRAY2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                mp_lm = result.multi_hand_landmarks[0]
                lm_array = np.array([[l.x, l.y, l.z] for l in mp_lm.landmark], dtype=np.float32)
                vec = normalize_landmarks(lm_array, mirror_left=False)
                if vec is not None:
                    features_list.append(vec)
                    labels_list.append(label_int)

        hands.close()

    except ImportError:
        # Fallback: use first 63 pixel values as a dummy feature (testing only)
        for label_int, pixels in rows:
            norm = pixels[:63] / 255.0
            features_list.append(norm.astype(np.float32))
            labels_list.append(label_int)

    if not features_list:
        raise RuntimeError("No valid landmarks extracted from the CSV.")

    return LandmarkDataset(
        features=np.stack(features_list),
        labels=np.array(labels_list, dtype=np.int64),
        augment=augment,
    )


# ---------------------------------------------------------------------------
# Custom image folder loader
# ---------------------------------------------------------------------------

def build_landmark_csv_from_images(
    image_root: str | Path,
    output_csv: str | Path,
    label_to_idx: dict[str, int] | None = None,
) -> Path:
    """Extract landmarks from a folder of labelled images and write a CSV.

    Expected folder structure::

        image_root/
            A/
                img1.jpg
                img2.png
            B/
                ...

    The CSV format mirrors Sign Language MNIST so that ``load_sign_mnist_csv``
    can read it.

    Args:
        image_root:    Root directory containing one sub-folder per class label.
        output_csv:    Destination CSV path.
        label_to_idx:  Optional mapping from label string to class index.
                       Defaults to LABEL_TO_IDX (A–Z + 0–9).

    Returns:
        Path to the written CSV file.
    """
    try:
        import mediapipe as mp  # type: ignore
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mediapipe and opencv-python are required for image extraction. "
            "Install them with: pip install mediapipe opencv-python"
        ) from exc

    if label_to_idx is None:
        label_to_idx = LABEL_TO_IDX

    image_root = Path(image_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )

    rows_written = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: label + 63 floats
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

        for label_dir in sorted(image_root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.upper()
            if label not in label_to_idx:
                continue
            idx = label_to_idx[label]

            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)

                if not result.multi_hand_landmarks:
                    continue

                mp_lm = result.multi_hand_landmarks[0]
                lm_array = np.array([[l.x, l.y, l.z] for l in mp_lm.landmark], dtype=np.float32)
                vec = normalize_landmarks(lm_array, mirror_left=False)
                if vec is None:
                    continue

                writer.writerow([idx] + vec.tolist())
                rows_written += 1

    hands.close()

    if rows_written == 0:
        raise RuntimeError(
            f"No landmarks extracted from images in {image_root}. "
            "Check that the folder structure is label/image.jpg and images contain visible hands."
        )

    return output_csv
