"""Benchmark module (M6)."""

from .config import (
    TeamComposition,
    ModelConfig,
    TeamAssignment,
    MatchupConfig,
    ExperimentConfig,
    generate_matchups,
    count_total_games,
)
from .model_farm import (
    ModelFarmFile,
    load_model_farm,
)
from .runner import (
    BenchmarkResult,
    BenchmarkProgress,
    BenchmarkRunner,
    run_benchmark,
)
from .leaderboard import (
    ConfidenceInterval,
    LeaderboardEntry,
    HeadToHeadEntry,
    Leaderboard,
    wilson_score_interval,
    standard_error,
    build_leaderboard,
    export_leaderboard_markdown,
)

__all__ = [
    # Config
    "TeamComposition",
    "ModelConfig",
    "TeamAssignment",
    "MatchupConfig",
    "ExperimentConfig",
    "generate_matchups",
    "count_total_games",
    # Model farm
    "ModelFarmFile",
    "load_model_farm",
    # Runner
    "BenchmarkResult",
    "BenchmarkProgress",
    "BenchmarkRunner",
    "run_benchmark",
    # Leaderboard
    "ConfidenceInterval",
    "LeaderboardEntry",
    "HeadToHeadEntry",
    "Leaderboard",
    "wilson_score_interval",
    "standard_error",
    "build_leaderboard",
    "export_leaderboard_markdown",
]
