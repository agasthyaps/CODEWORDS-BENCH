"""Event types for benchmark SSE streaming."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of benchmark events."""

    GAME_START = "game_start"
    GAME_COMPLETE = "game_complete"
    GAME_ERROR = "game_error"
    PROGRESS = "progress"
    FINDING = "finding"
    BENCHMARK_COMPLETE = "benchmark_complete"
    BENCHMARK_PAUSED = "benchmark_paused"
    BENCHMARK_ERROR = "benchmark_error"


class GameStartEvent(BaseModel):
    """Emitted when a game starts."""

    game_type: Literal["codenames", "decrypto", "hanabi"]
    game_id: str
    seed: int
    matchup_id: str | None = None
    models: dict[str, str]  # role/position -> model_id


class GameCompleteEvent(BaseModel):
    """Emitted when a game completes successfully."""

    game_type: Literal["codenames", "decrypto", "hanabi"]
    game_id: str
    seed: int
    matchup_id: str | None = None
    result: dict[str, Any]  # winner, score, etc.
    duration_seconds: float


class GameErrorEvent(BaseModel):
    """Emitted when a game fails."""

    game_type: Literal["codenames", "decrypto", "hanabi"]
    game_id: str
    seed: int
    matchup_id: str | None = None
    error: str
    attempt: int


class GameTypeProgressData(BaseModel):
    """Progress data for a single game type."""

    total: int
    completed: int
    failed: int
    running: int


class ProgressEvent(BaseModel):
    """Emitted periodically with overall progress."""

    codenames: GameTypeProgressData | None = None
    decrypto: GameTypeProgressData | None = None
    hanabi: GameTypeProgressData | None = None
    elapsed_seconds: float


class FindingEvent(BaseModel):
    """Emitted when interim analysis produces a finding."""

    finding_id: str
    game_type: Literal["codenames", "decrypto", "hanabi"]
    games_analyzed: int
    preview: str  # First ~200 chars of analysis


class BenchmarkCompleteEvent(BaseModel):
    """Emitted when the entire benchmark completes."""

    experiment_name: str
    total_games: int
    completed_games: int
    failed_games: int
    elapsed_seconds: float
    findings_count: int


class BenchmarkPausedEvent(BaseModel):
    """Emitted when the benchmark is paused."""

    experiment_name: str
    completed_games: int
    remaining_games: int


class BenchmarkErrorEvent(BaseModel):
    """Emitted when the benchmark encounters a fatal error."""

    error: str
    recoverable: bool


class BenchmarkEvent(BaseModel):
    """Wrapper for all benchmark events."""

    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: (
        GameStartEvent
        | GameCompleteEvent
        | GameErrorEvent
        | ProgressEvent
        | FindingEvent
        | BenchmarkCompleteEvent
        | BenchmarkPausedEvent
        | BenchmarkErrorEvent
    )

    @classmethod
    def game_start(
        cls,
        game_type: Literal["codenames", "decrypto", "hanabi"],
        game_id: str,
        seed: int,
        models: dict[str, str],
        matchup_id: str | None = None,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.GAME_START,
            data=GameStartEvent(
                game_type=game_type,
                game_id=game_id,
                seed=seed,
                matchup_id=matchup_id,
                models=models,
            ),
        )

    @classmethod
    def game_complete(
        cls,
        game_type: Literal["codenames", "decrypto", "hanabi"],
        game_id: str,
        seed: int,
        result: dict[str, Any],
        duration_seconds: float,
        matchup_id: str | None = None,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.GAME_COMPLETE,
            data=GameCompleteEvent(
                game_type=game_type,
                game_id=game_id,
                seed=seed,
                matchup_id=matchup_id,
                result=result,
                duration_seconds=duration_seconds,
            ),
        )

    @classmethod
    def game_error(
        cls,
        game_type: Literal["codenames", "decrypto", "hanabi"],
        game_id: str,
        seed: int,
        error: str,
        attempt: int,
        matchup_id: str | None = None,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.GAME_ERROR,
            data=GameErrorEvent(
                game_type=game_type,
                game_id=game_id,
                seed=seed,
                matchup_id=matchup_id,
                error=error,
                attempt=attempt,
            ),
        )

    @classmethod
    def progress(
        cls,
        codenames: GameTypeProgressData | None,
        decrypto: GameTypeProgressData | None,
        hanabi: GameTypeProgressData | None,
        elapsed_seconds: float,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.PROGRESS,
            data=ProgressEvent(
                codenames=codenames,
                decrypto=decrypto,
                hanabi=hanabi,
                elapsed_seconds=elapsed_seconds,
            ),
        )

    @classmethod
    def finding(
        cls,
        finding_id: str,
        game_type: Literal["codenames", "decrypto", "hanabi"],
        games_analyzed: int,
        preview: str,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.FINDING,
            data=FindingEvent(
                finding_id=finding_id,
                game_type=game_type,
                games_analyzed=games_analyzed,
                preview=preview,
            ),
        )

    @classmethod
    def benchmark_complete(
        cls,
        experiment_name: str,
        total_games: int,
        completed_games: int,
        failed_games: int,
        elapsed_seconds: float,
        findings_count: int,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.BENCHMARK_COMPLETE,
            data=BenchmarkCompleteEvent(
                experiment_name=experiment_name,
                total_games=total_games,
                completed_games=completed_games,
                failed_games=failed_games,
                elapsed_seconds=elapsed_seconds,
                findings_count=findings_count,
            ),
        )

    @classmethod
    def benchmark_paused(
        cls,
        experiment_name: str,
        completed_games: int,
        remaining_games: int,
    ) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.BENCHMARK_PAUSED,
            data=BenchmarkPausedEvent(
                experiment_name=experiment_name,
                completed_games=completed_games,
                remaining_games=remaining_games,
            ),
        )

    @classmethod
    def benchmark_error(cls, error: str, recoverable: bool = False) -> "BenchmarkEvent":
        return cls(
            event_type=EventType.BENCHMARK_ERROR,
            data=BenchmarkErrorEvent(error=error, recoverable=recoverable),
        )
