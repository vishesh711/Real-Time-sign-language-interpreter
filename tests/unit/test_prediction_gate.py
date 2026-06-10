"""Unit tests for PredictionGate temporal filter."""

from __future__ import annotations

from utils.gate import PredictionGate


class TestPredictionGate:
    def test_stable_label_commits_once(self):
        gate = PredictionGate(vote_window=3, hold_frames=3, cooldown_frames=10)
        commits = []
        for _ in range(10):
            result = gate.update("A", confidence=0.95)
            if result is not None:
                commits.append(result)
        assert commits == ["A"]

    def test_cooldown_prevents_immediate_recommit(self):
        gate = PredictionGate(vote_window=3, hold_frames=3, cooldown_frames=10)
        first_commit_at: int | None = None
        commits_during_cooldown: list[str] = []

        for i in range(50):
            result = gate.update("B", confidence=0.95)
            if result is None:
                continue
            if first_commit_at is None:
                first_commit_at = i
            elif i <= first_commit_at + 10:
                commits_during_cooldown.append(result)

        assert first_commit_at is not None
        assert commits_during_cooldown == []

    def test_low_confidence_frames_do_not_commit(self):
        gate = PredictionGate(vote_window=3, hold_frames=3, confidence_threshold=0.8)
        commits = []
        for _ in range(20):
            result = gate.update("C", confidence=0.5)
            if result is not None:
                commits.append(result)
        assert commits == []

    def test_reset_clears_state(self):
        gate = PredictionGate(vote_window=3, hold_frames=3, cooldown_frames=10)
        for _ in range(5):
            gate.update("D", confidence=0.95)
        gate.reset()
        commits = []
        for _ in range(10):
            result = gate.update("D", confidence=0.95)
            if result is not None:
                commits.append(result)
        assert commits == ["D"]

    def test_transient_label_suppressed(self):
        """A label that appears briefly then changes should not commit."""
        gate = PredictionGate(vote_window=5, hold_frames=5, cooldown_frames=10)
        commits = []
        # Brief 'E' run (2 frames) — too short to reach hold_frames=5
        for _ in range(2):
            gate.update("E", confidence=0.95)
        # Switch to 'F' for remaining frames
        for _ in range(20):
            result = gate.update("F", confidence=0.95)
            if result is not None:
                commits.append(result)
        assert "E" not in commits
        assert commits == ["F"]
