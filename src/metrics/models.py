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
    # v1.1: prediction-based ToM (falls back to clue_efficiency when unavailable)
    theory_of_mind_score: float

    # Optional submetrics when prediction traces are available
    tom_predictions_count: int = 0
    # Mind-modeling (ToM) core metric: overlap@k (recall@k) between predicted and actual guesses.
    tom_overlap_at_k: float | None = None
    tom_translated_overlap_at_k: float | None = None
    tom_rank_correlation: float | None = None
    tom_confusion_calibration: float | None = None

    # Compliance (separate from ToM)
    tom_format_compliance_rate: float | None = None  # fraction of turns with parseable prediction output
    tom_boardword_compliance_rate: float | None = None  # fraction of turns where top-k predictions are all board words
    tom_non_board_rate_top_k: float | None = None  # average fraction of non-board words in top-k predictions

    # Risk / calibration
    cluer_confidence_mean: float | None = None  # 1-5
    cluer_overconfidence_rate: float | None = None  # high confidence + low overlap@k
    guesser_confidence_mean: float | None = None  # 1-5
    guesser_overconfidence_rate: float | None = None  # high confidence + any wrong guess that turn
    guesser_confidence_correctness_n: int = 0
    guesser_confidence_correctness_point_biserial: float | None = None
    guesser_confidence_correctness_spearman: float | None = None


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
