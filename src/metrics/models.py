"""Data models for metrics collection (M5)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.engine import Team, GameConfig


class TeamMetrics(BaseModel):
    """Metrics for a single team in an episode."""
    team: Team
    words_cleared: int
    assassin_hit: bool

    # Clue metrics
    total_clues: int
    avg_clue_number: float  # Average clue number (ambition)
    clue_efficiency: float  # correct_guesses / sum(clue_numbers)

    # Guess metrics
    total_guesses: int
    correct_guesses: int
    wrong_guesses: int  # Opponent cards + neutral cards
    opponent_guesses: int  # Specifically opponent cards
    neutral_guesses: int  # Specifically neutral cards
    guess_accuracy: float  # correct / total

    # Discussion metrics
    avg_discussion_rounds: float
    consensus_rate: float  # % of turns with explicit consensus
    avg_discussion_length: int  # Total chars across discussions

    # Theory of Mind score (fallback: correct_guesses / sum(clue_numbers))
    # TODO: Implement structured extraction from REASONING field
    theory_of_mind_score: float


class EpisodeMetrics(BaseModel):
    """Metrics for a complete episode."""
    episode_id: str
    winner: Team | None
    turns_to_win: int

    # Per team metrics
    red_metrics: TeamMetrics
    blue_metrics: TeamMetrics

    # Derived scores
    red_coordination_score: float
    blue_coordination_score: float


class AggregateMetrics(BaseModel):
    """Aggregate metrics across multiple episodes."""
    config: GameConfig | None = None
    episodes: int

    win_rate_red: float
    win_rate_blue: float
    draw_rate: float  # Games with no winner (max turns reached)

    avg_turns_to_win: float
    std_turns_to_win: float

    avg_coordination_score_red: float
    avg_coordination_score_blue: float

    avg_theory_of_mind_red: float
    avg_theory_of_mind_blue: float

    assassin_rate: float  # % of games ending in assassin hit

    # Per-metric averages (useful for detailed analysis)
    avg_clue_efficiency_red: float = 0.0
    avg_clue_efficiency_blue: float = 0.0
    avg_guess_accuracy_red: float = 0.0
    avg_guess_accuracy_blue: float = 0.0
    avg_consensus_rate_red: float = 0.0
    avg_consensus_rate_blue: float = 0.0
