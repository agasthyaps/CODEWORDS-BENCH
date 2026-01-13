"""Metrics module (M5)."""

from .models import TeamMetrics, EpisodeMetrics, AggregateMetrics
from .collector import (
    compute_team_metrics,
    compute_coordination_score,
    compute_episode_metrics,
    compute_aggregate_metrics,
)
from .export import (
    export_episode_metrics_json,
    export_aggregate_metrics_json,
    export_episodes_csv,
    export_aggregate_csv,
    export_episode_markdown,
    export_aggregate_markdown,
    export_metrics,
)

__all__ = [
    # Models
    "TeamMetrics",
    "EpisodeMetrics",
    "AggregateMetrics",
    # Collector
    "compute_team_metrics",
    "compute_coordination_score",
    "compute_episode_metrics",
    "compute_aggregate_metrics",
    # Export
    "export_episode_metrics_json",
    "export_aggregate_metrics_json",
    "export_episodes_csv",
    "export_aggregate_csv",
    "export_episode_markdown",
    "export_aggregate_markdown",
    "export_metrics",
]
