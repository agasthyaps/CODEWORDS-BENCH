"""Benchmark state management with crash recovery."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import CloudBenchmarkConfig, get_data_dir


class GameTypeProgress(BaseModel):
    """Progress tracking for a single game type."""

    total_games: int = 0
    completed_games: int = 0
    failed_games: int = 0
    running_games: int = 0
    last_analyzed_count: int = 0  # Games at last interim analysis


class BenchmarkState(BaseModel):
    """
    Persistent state for crash recovery.

    Saved atomically after each game completes.
    """

    experiment_name: str
    status: Literal["running", "paused", "complete", "error"] = "running"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Per-game-type progress
    codenames: GameTypeProgress = Field(default_factory=GameTypeProgress)
    decrypto: GameTypeProgress = Field(default_factory=GameTypeProgress)
    hanabi: GameTypeProgress = Field(default_factory=GameTypeProgress)

    # Completed game keys for crash recovery
    # Format: "game_type|matchup_id|seed|game_index" or "hanabi|model_combo|seed"
    completed_keys: set[str] = Field(default_factory=set)

    # Error tracking for circuit breaker
    consecutive_failures: int = 0
    last_error: str | None = None

    # Analysis tracking
    findings_count: int = 0

    # Config snapshot for resume validation
    config_snapshot: dict | None = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def get_state_path(cls, experiment_name: str) -> Path:
        """Get path to state file for an experiment."""
        return get_data_dir() / experiment_name / "benchmark_state.json"

    @classmethod
    def load(cls, experiment_name: str) -> "BenchmarkState | None":
        """Load state from disk if it exists."""
        path = cls.get_state_path(experiment_name)
        if not path.exists():
            return None

        with open(path, "r") as f:
            data = json.load(f)

        # Convert completed_keys back to set
        data["completed_keys"] = set(data.get("completed_keys", []))
        return cls.model_validate(data)

    @classmethod
    def load_or_create(
        cls, config: CloudBenchmarkConfig
    ) -> "BenchmarkState":
        """Load existing state or create new one."""
        existing = cls.load(config.experiment_name)
        if existing:
            # Validate config compatibility
            if existing.config_snapshot:
                # Basic check - model_ids should match
                if existing.config_snapshot.get("model_ids") != config.model_ids:
                    raise ValueError(
                        "Cannot resume: model_ids changed. "
                        "Use a new experiment name or delete the existing state."
                    )
            return existing

        # Create new state
        state = cls(
            experiment_name=config.experiment_name,
            config_snapshot=config.model_dump(mode="json"),
        )

        # Initialize progress totals (will be set by runner)
        return state

    def save(self) -> None:
        """Save state atomically using temp file + rename."""
        self.updated_at = datetime.utcnow()
        path = self.get_state_path(self.experiment_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        # Convert set to list for JSON
        data["completed_keys"] = list(self.completed_keys)

        # Atomic write: write to temp file, then rename
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as f:
            json.dump(data, f, indent=2, default=str)
            temp_path = Path(f.name)

        temp_path.rename(path)

    def get_progress_key(
        self,
        game_type: str,
        matchup_id: str,
        seed: int,
        game_index: int = 0,
    ) -> str:
        """Generate unique key for a game configuration."""
        return f"{game_type}|{matchup_id}|{seed}|{game_index}"

    def is_completed(
        self,
        game_type: str,
        matchup_id: str,
        seed: int,
        game_index: int = 0,
    ) -> bool:
        """Check if a game has been completed."""
        key = self.get_progress_key(game_type, matchup_id, seed, game_index)
        return key in self.completed_keys

    def mark_completed(
        self,
        game_type: str,
        matchup_id: str,
        seed: int,
        game_index: int = 0,
    ) -> None:
        """Mark a game as completed."""
        key = self.get_progress_key(game_type, matchup_id, seed, game_index)
        self.completed_keys.add(key)

        # Update progress for game type
        progress = getattr(self, game_type)
        progress.completed_games += 1

        # Reset consecutive failures on success
        self.consecutive_failures = 0

    def mark_failed(self, game_type: str, error: str) -> None:
        """Mark a game as failed."""
        progress = getattr(self, game_type)
        progress.failed_games += 1
        self.consecutive_failures += 1
        self.last_error = error

    def mark_running(self, game_type: str, delta: int = 1) -> None:
        """Update running count for a game type."""
        progress = getattr(self, game_type)
        progress.running_games += delta

    def should_circuit_break(self, threshold: int) -> bool:
        """Check if we should stop due to too many consecutive failures."""
        return self.consecutive_failures >= threshold

    def total_completed(self) -> int:
        """Get total completed games across all types."""
        return (
            self.codenames.completed_games
            + self.decrypto.completed_games
            + self.hanabi.completed_games
        )

    def total_failed(self) -> int:
        """Get total failed games across all types."""
        return (
            self.codenames.failed_games
            + self.decrypto.failed_games
            + self.hanabi.failed_games
        )

    def total_remaining(self) -> int:
        """Get total remaining games across all types."""
        return (
            (self.codenames.total_games - self.codenames.completed_games - self.codenames.failed_games)
            + (self.decrypto.total_games - self.decrypto.completed_games - self.decrypto.failed_games)
            + (self.hanabi.total_games - self.hanabi.completed_games - self.hanabi.failed_games)
        )
