"""Benchmark module (M6)."""

from .config import (
    TeamComposition,
    ModelConfig,
    TeamAssignment,
    MatchupConfig,
    ExperimentConfig,
    generate_matchups,
    count_total_games,
    select_family_representatives,
    sparse_family_matchup_subset,
)
from .protocol import (
    BenchmarkCondition,
    BenchmarkConditionName,
    HarnessEvent,
    BenchmarkPenalty,
    RunManifest,
    default_benchmark_condition,
    resolve_condition,
    build_run_manifest,
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
    MetricEvidence,
    ReportedMetric,
    ToMBlock,
    RobustnessEntry,
    EntryEvidence,
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
    "select_family_representatives",
    "sparse_family_matchup_subset",
    "BenchmarkCondition",
    "BenchmarkConditionName",
    "HarnessEvent",
    "BenchmarkPenalty",
    "RunManifest",
    "default_benchmark_condition",
    "resolve_condition",
    "build_run_manifest",
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
    "MetricEvidence",
    "ReportedMetric",
    "ToMBlock",
    "RobustnessEntry",
    "EntryEvidence",
    "LeaderboardEntry",
    "HeadToHeadEntry",
    "Leaderboard",
    "wilson_score_interval",
    "standard_error",
    "build_leaderboard",
    "export_leaderboard_markdown",
]
