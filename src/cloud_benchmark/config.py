"""Configuration for cloud benchmark runs."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from src.engine import GameMode


def get_data_dir() -> Path:
    """Get benchmark data directory from env or default."""
    env_dir = os.environ.get("BENCHMARK_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("benchmark_results")


class CloudBenchmarkConfig(BaseModel):
    """Configuration for a cloud benchmark run."""

    model_config = {"protected_namespaces": ()}

    experiment_name: str

    # Models to benchmark (by model_id from models.json)
    model_ids: list[str]

    # Seeds
    seed_count: int = 30
    seed_list: list[int] | None = None  # Override with specific seeds

    # Game type toggles
    run_codenames: bool = True
    run_decrypto: bool = True
    run_hanabi: bool = True

    # Concurrency (games running simultaneously per type)
    codenames_concurrency: int = 2
    decrypto_concurrency: int = 2
    hanabi_concurrency: int = 1

    # Game-specific settings
    codenames_mode: GameMode = GameMode.STANDARD
    codenames_max_turns: int = 50
    codenames_max_discussion_rounds: int = 3
    decrypto_max_rounds: int = 8
    decrypto_max_discussion_turns: int = 2

    # Analysis
    interim_analysis_batch_size: int = 10

    # Error handling
    max_retries: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0
    max_consecutive_failures: int = 5  # Circuit breaker threshold
    temperature: float = 0.7

    # Output directory (configurable via env var)
    output_dir: str = Field(default_factory=lambda: str(get_data_dir()))

    def get_seeds(self) -> list[int]:
        """Get list of seeds to use."""
        if self.seed_list:
            return self.seed_list
        return list(range(self.seed_count))

    def get_output_path(self) -> Path:
        """Get full output path for this experiment."""
        return Path(self.output_dir) / self.experiment_name
