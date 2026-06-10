"""
Server configuration using pydantic-settings.

All settings can be overridden via environment variables or a `.env` file
in the working directory.  Variable names match the field names (case-
insensitive by default in pydantic-settings).

Requirements: 8.1
"""

from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Model paths
    # ------------------------------------------------------------------
    fingerspell_onnx_path: str = Field(
        default="models/fingerspell.onnx",
        description="Path to the fingerspelling ONNX model file.",
    )
    word_onnx_path: str = Field(
        default="models/word.onnx",
        description="Path to the word-level ONNX model file.",
    )
    labels_json_path: str = Field(
        default="models/labels.json",
        description="Path to the labels JSON sidecar file.",
    )

    # ------------------------------------------------------------------
    # Confidence thresholds
    # ------------------------------------------------------------------
    detection_conf_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum MediaPipe detection confidence to pass landmarks.",
    )
    classifier_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum classifier confidence for a valid prediction.",
    )

    # ------------------------------------------------------------------
    # PredictionGate parameters
    # ------------------------------------------------------------------
    vote_window: int = Field(
        default=7,
        gt=0,
        description="Sliding-window size for majority vote (frames).",
    )
    hold_frames: int = Field(
        default=12,
        gt=0,
        description="Number of consecutive stable frames required before commit.",
    )
    cooldown_frames: int = Field(
        default=20,
        ge=0,
        description="Suppression gap (frames) between two accepted predictions.",
    )
    temperature: float = Field(
        default=0.8,
        gt=0.0,
        description="Temperature for softmax scaling (< 1 sharpens, > 1 flattens).",
    )

    # ------------------------------------------------------------------
    # Word endpoint parameters
    # ------------------------------------------------------------------
    seq_len: int = Field(
        default=30,
        gt=0,
        description="Fixed temporal window the word model expects (frames).",
    )
    word_stride: int = Field(
        default=5,
        gt=0,
        description="Run word-level inference every N frames.",
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins.",
    )
