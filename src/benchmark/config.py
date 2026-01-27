"""Experiment configuration for benchmark runs (M6)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.engine import Team, GameMode

# Clue generation mode for benchmark experiments
ClueGenerationMode = Literal["standard", "deliberate"]


class TeamComposition(str, Enum):
    """Team composition types for experiments."""
    HOMOGENEOUS = "homogeneous"  # All 6 agents same model
    MIXED_CLUER = "mixed_cluer"  # Cluers are model A, guessers are model B
    MIXED_GUESSER = "mixed_guesser"  # Cluer + guesser_1 are model A, guesser_2 is model B
    HETEROGENEOUS = "heterogeneous"  # All different (if 3+ models)


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    model_config = {"protected_namespaces": ()}

    name: str  # Friendly name for display
    model_id: str  # Actual model ID for API calls
    provider: str = "openrouter"  # openrouter, openai, anthropic
    base_url: str | None = None  # Provider base URL override (primarily OpenRouter)


class TeamAssignment(BaseModel):
    """Model assignments for a team."""
    cluer: ModelConfig
    guesser_1: ModelConfig
    guesser_2: ModelConfig


class MatchupConfig(BaseModel):
    """Configuration for a single matchup (red vs blue)."""
    red_team: TeamAssignment
    blue_team: TeamAssignment
    composition: TeamComposition
    # v1.1 metadata for pairing and analysis
    pair_key: str | None = None  # stable identifier for the unordered model pair
    config_type: str | None = None  # homog-A, homog-B, mixed-A-clue, mixed-B-clue
    direction: str | None = None  # A_RED_B_BLUE or A_BLUE_B_RED (relative to canonical A/B)


class ExperimentConfig(BaseModel):
    """Configuration for a benchmark experiment."""
    name: str
    description: str = ""

    # Models to test
    models: list[ModelConfig]

    # Matchup selection strategy
    matchup_strategy: Literal["round_robin", "subset"] = "round_robin"
    matchup_subset: list[tuple[str, str]] | None = None  # pairs of model_id

    # Game configurations
    game_modes: list[GameMode] = Field(default_factory=lambda: [GameMode.STANDARD])

    # Team compositions to test
    team_compositions: list[TeamComposition] = Field(
        default_factory=lambda: [TeamComposition.HOMOGENEOUS]
    )

    # Seeds for reproducibility
    seeds: list[int] = Field(default_factory=lambda: list(range(100)))

    # Run settings
    games_per_config: int = 1  # Games per (matchup, mode, seed) combination
    temperature: float = 0.7
    max_retries: int = 3
    max_discussion_rounds: int = 3
    max_turns: int = 50
    clue_generation_mode: ClueGenerationMode = "standard"

    # Error handling
    max_consecutive_failures: int = 5  # Skip config after this many failures
    retry_delay_base: float = 1.0  # Base delay for exponential backoff (seconds)
    retry_delay_max: float = 60.0  # Max delay between retries

    # Output
    output_dir: str = "benchmark_results"


def generate_matchups(config: ExperimentConfig) -> list[MatchupConfig]:
    """
    Generate all matchup configurations for an experiment.

    v1.1: For each model pair (A,B), generate 4 configs and run both directions.
      - homog-A:  RED = A,A,A   BLUE = B,B,B
      - homog-B:  RED = B,B,B   BLUE = A,A,A
      - mixed-A-clue: RED = (A,B,B) BLUE = (B,A,A)
      - mixed-B-clue: RED = (B,A,A) BLUE = (A,B,B)
    Then, for each config, emit the side-swapped direction by swapping red/blue.

    Args:
        config: Experiment configuration

    Returns:
        List of MatchupConfig
    """
    models = config.models
    model_by_id = {m.model_id: m for m in models}

    def _canonical_pair(a_id: str, b_id: str) -> tuple[ModelConfig, ModelConfig, str]:
        """Return (A,B,pair_key) where A/B are canonical (lexicographic by model_id)."""
        if a_id == b_id:
            raise ValueError("Pair must contain two distinct model_ids")
        a_model = model_by_id[a_id]
        b_model = model_by_id[b_id]
        if a_model.model_id <= b_model.model_id:
            A, B = a_model, b_model
        else:
            A, B = b_model, a_model
        return A, B, f"{A.model_id}|{B.model_id}"

    # Determine which unordered model pairs to run
    pairs: list[tuple[str, str]] = []
    if config.matchup_strategy == "subset":
        if not config.matchup_subset:
            raise ValueError("matchup_strategy=subset requires matchup_subset")
        pairs = list(config.matchup_subset)
    else:
        # Round-robin unordered pairs (no mirrors)
        model_ids = sorted(model_by_id.keys())
        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                pairs.append((model_ids[i], model_ids[j]))

    matchups: list[MatchupConfig] = []

    for a_id, b_id in pairs:
        A, B, pair_key = _canonical_pair(a_id, b_id)

        def _direction_for(red_team: TeamAssignment) -> str:
            # Interpret direction relative to canonical A/B based on which model is the RED cluer.
            return "A_RED_B_BLUE" if red_team.cluer.model_id == A.model_id else "A_BLUE_B_RED"

        def _add_both_directions(base: MatchupConfig) -> None:
            # BASE
            base = base.model_copy(deep=True)
            base.pair_key = pair_key
            base.direction = _direction_for(base.red_team)
            matchups.append(base)

            # SWAPPED (swap teams)
            swapped = base.model_copy(deep=True)
            swapped.red_team, swapped.blue_team = swapped.blue_team, swapped.red_team
            swapped.direction = _direction_for(swapped.red_team)
            matchups.append(swapped)

        # 1) homog-A
        _add_both_directions(
            MatchupConfig(
                red_team=TeamAssignment(cluer=A, guesser_1=A, guesser_2=A),
                blue_team=TeamAssignment(cluer=B, guesser_1=B, guesser_2=B),
                composition=TeamComposition.HOMOGENEOUS,
                config_type="homog-A",
            )
        )

        # 2) homog-B
        _add_both_directions(
            MatchupConfig(
                red_team=TeamAssignment(cluer=B, guesser_1=B, guesser_2=B),
                blue_team=TeamAssignment(cluer=A, guesser_1=A, guesser_2=A),
                composition=TeamComposition.HOMOGENEOUS,
                config_type="homog-B",
            )
        )

        # 3) mixed-A-clue
        _add_both_directions(
            MatchupConfig(
                red_team=TeamAssignment(cluer=A, guesser_1=B, guesser_2=B),
                blue_team=TeamAssignment(cluer=B, guesser_1=A, guesser_2=A),
                composition=TeamComposition.MIXED_CLUER,
                config_type="mixed-A-clue",
            )
        )

        # 4) mixed-B-clue
        _add_both_directions(
            MatchupConfig(
                red_team=TeamAssignment(cluer=B, guesser_1=A, guesser_2=A),
                blue_team=TeamAssignment(cluer=A, guesser_1=B, guesser_2=B),
                composition=TeamComposition.MIXED_CLUER,
                config_type="mixed-B-clue",
            )
        )

    return matchups


def count_total_games(config: ExperimentConfig) -> int:
    """Count total number of games in an experiment."""
    matchups = generate_matchups(config)
    return (
        len(matchups) *
        len(config.game_modes) *
        len(config.seeds) *
        config.games_per_config
    )
