"""Experiment configuration for benchmark runs (M6)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.engine import Team, GameMode


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


class ExperimentConfig(BaseModel):
    """Configuration for a benchmark experiment."""
    name: str
    description: str = ""

    # Models to test
    models: list[ModelConfig]

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

    # Error handling
    max_consecutive_failures: int = 5  # Skip config after this many failures
    retry_delay_base: float = 1.0  # Base delay for exponential backoff (seconds)
    retry_delay_max: float = 60.0  # Max delay between retries

    # Output
    output_dir: str = "benchmark_results"


def generate_matchups(config: ExperimentConfig) -> list[MatchupConfig]:
    """
    Generate all matchup configurations for an experiment.

    For homogeneous: each model plays against each other model
    For mixed compositions: various model combinations

    Args:
        config: Experiment configuration

    Returns:
        List of MatchupConfig
    """
    matchups = []
    models = config.models

    for composition in config.team_compositions:
        if composition == TeamComposition.HOMOGENEOUS:
            # Each model plays against each other (including mirror matches)
            for i, model_a in enumerate(models):
                for model_b in models[i:]:  # Avoid duplicate matchups
                    matchups.append(MatchupConfig(
                        red_team=TeamAssignment(
                            cluer=model_a,
                            guesser_1=model_a,
                            guesser_2=model_a,
                        ),
                        blue_team=TeamAssignment(
                            cluer=model_b,
                            guesser_1=model_b,
                            guesser_2=model_b,
                        ),
                        composition=composition,
                    ))

        elif composition == TeamComposition.MIXED_CLUER:
            # Cluers are model A, guessers are model B
            if len(models) >= 2:
                for model_a in models:
                    for model_b in models:
                        if model_a != model_b:
                            matchups.append(MatchupConfig(
                                red_team=TeamAssignment(
                                    cluer=model_a,
                                    guesser_1=model_b,
                                    guesser_2=model_b,
                                ),
                                blue_team=TeamAssignment(
                                    cluer=model_a,
                                    guesser_1=model_b,
                                    guesser_2=model_b,
                                ),
                                composition=composition,
                            ))

        elif composition == TeamComposition.MIXED_GUESSER:
            # Cluer + guesser_1 are model A, guesser_2 is model B
            if len(models) >= 2:
                for model_a in models:
                    for model_b in models:
                        if model_a != model_b:
                            matchups.append(MatchupConfig(
                                red_team=TeamAssignment(
                                    cluer=model_a,
                                    guesser_1=model_a,
                                    guesser_2=model_b,
                                ),
                                blue_team=TeamAssignment(
                                    cluer=model_a,
                                    guesser_1=model_a,
                                    guesser_2=model_b,
                                ),
                                composition=composition,
                            ))

        elif composition == TeamComposition.HETEROGENEOUS:
            # All different models (needs 3+ models)
            if len(models) >= 3:
                for i, model_a in enumerate(models):
                    for j, model_b in enumerate(models):
                        for k, model_c in enumerate(models):
                            if i != j and j != k and i != k:
                                matchups.append(MatchupConfig(
                                    red_team=TeamAssignment(
                                        cluer=model_a,
                                        guesser_1=model_b,
                                        guesser_2=model_c,
                                    ),
                                    blue_team=TeamAssignment(
                                        cluer=model_a,
                                        guesser_1=model_b,
                                        guesser_2=model_c,
                                    ),
                                    composition=composition,
                                ))

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
