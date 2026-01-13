"""Leaderboard and statistical analysis (M6)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from src.engine import Team, GameMode

from .runner import BenchmarkResult


class ConfidenceInterval(BaseModel):
    """95% confidence interval."""
    lower: float
    upper: float


class LeaderboardEntry(BaseModel):
    """Entry in the leaderboard."""
    model: str
    role: Literal["overall", "cluer", "guesser"] = "overall"
    games: int
    wins: int
    win_rate: float
    win_rate_ci: ConfidenceInterval
    avg_coordination_score: float
    avg_clue_efficiency: float | None = None  # Cluer only
    avg_guess_accuracy: float | None = None  # Guesser only
    avg_theory_of_mind: float


class HeadToHeadEntry(BaseModel):
    """Head-to-head comparison between two models."""
    model_config = {"protected_namespaces": ()}

    model_a: str
    model_b: str
    games: int
    model_a_wins: int
    model_b_wins: int
    draws: int
    model_a_win_rate: float
    model_a_win_rate_ci: ConfidenceInterval


class Leaderboard(BaseModel):
    """Complete leaderboard with various views."""
    overall: list[LeaderboardEntry] = Field(default_factory=list)
    by_cluer: list[LeaderboardEntry] = Field(default_factory=list)
    by_guesser: list[LeaderboardEntry] = Field(default_factory=list)
    by_mode: dict[str, list[LeaderboardEntry]] = Field(default_factory=dict)
    head_to_head: list[HeadToHeadEntry] = Field(default_factory=list)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """
    Calculate Wilson score confidence interval for a proportion.

    This is more accurate than the normal approximation for small samples
    and proportions near 0 or 1.

    Args:
        successes: Number of successes
        total: Total trials
        confidence: Confidence level (default 0.95)

    Returns:
        ConfidenceInterval with lower and upper bounds
    """
    if total == 0:
        return ConfidenceInterval(lower=0.0, upper=1.0)

    # Z-score for 95% confidence
    z = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 2.576

    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return ConfidenceInterval(lower=lower, upper=upper)


def standard_error(values: list[float]) -> float:
    """Calculate standard error of the mean."""
    if len(values) < 2:
        return 0.0

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance / n)


def _extract_model_stats(
    results: list[BenchmarkResult],
    model_id: str,
    role: Literal["overall", "cluer", "guesser"] = "overall",
) -> dict:
    """Extract statistics for a specific model."""
    games = 0
    wins = 0
    coordination_scores = []
    clue_efficiencies = []
    guess_accuracies = []
    tom_scores = []

    for result in results:
        if result.error or result.metrics is None:
            continue

        # Check if model played in this game
        red_match = False
        blue_match = False

        if role == "overall":
            red_match = model_id in result.red_models.values()
            blue_match = model_id in result.blue_models.values()
        elif role == "cluer":
            red_match = result.red_models.get("cluer") == model_id
            blue_match = result.blue_models.get("cluer") == model_id
        elif role == "guesser":
            red_match = (
                result.red_models.get("guesser_1") == model_id or
                result.red_models.get("guesser_2") == model_id
            )
            blue_match = (
                result.blue_models.get("guesser_1") == model_id or
                result.blue_models.get("guesser_2") == model_id
            )

        if red_match:
            games += 1
            if result.winner == Team.RED:
                wins += 1
            coordination_scores.append(result.metrics.red_coordination_score)
            clue_efficiencies.append(result.metrics.red_metrics.clue_efficiency)
            guess_accuracies.append(result.metrics.red_metrics.guess_accuracy)
            tom_scores.append(result.metrics.red_metrics.theory_of_mind_score)

        if blue_match:
            games += 1
            if result.winner == Team.BLUE:
                wins += 1
            coordination_scores.append(result.metrics.blue_coordination_score)
            clue_efficiencies.append(result.metrics.blue_metrics.clue_efficiency)
            guess_accuracies.append(result.metrics.blue_metrics.guess_accuracy)
            tom_scores.append(result.metrics.blue_metrics.theory_of_mind_score)

    return {
        "games": games,
        "wins": wins,
        "coordination_scores": coordination_scores,
        "clue_efficiencies": clue_efficiencies,
        "guess_accuracies": guess_accuracies,
        "tom_scores": tom_scores,
    }


def _create_leaderboard_entry(
    model_id: str,
    stats: dict,
    role: Literal["overall", "cluer", "guesser"] = "overall",
) -> LeaderboardEntry:
    """Create a leaderboard entry from stats."""
    games = stats["games"]
    wins = stats["wins"]

    if games == 0:
        return LeaderboardEntry(
            model=model_id,
            role=role,
            games=0,
            wins=0,
            win_rate=0.0,
            win_rate_ci=ConfidenceInterval(lower=0.0, upper=1.0),
            avg_coordination_score=0.0,
            avg_theory_of_mind=0.0,
        )

    win_rate = wins / games
    win_rate_ci = wilson_score_interval(wins, games)

    avg_coord = (
        sum(stats["coordination_scores"]) / len(stats["coordination_scores"])
        if stats["coordination_scores"] else 0.0
    )
    avg_tom = (
        sum(stats["tom_scores"]) / len(stats["tom_scores"])
        if stats["tom_scores"] else 0.0
    )

    entry = LeaderboardEntry(
        model=model_id,
        role=role,
        games=games,
        wins=wins,
        win_rate=win_rate,
        win_rate_ci=win_rate_ci,
        avg_coordination_score=avg_coord,
        avg_theory_of_mind=avg_tom,
    )

    if role in ("overall", "cluer") and stats["clue_efficiencies"]:
        entry.avg_clue_efficiency = (
            sum(stats["clue_efficiencies"]) / len(stats["clue_efficiencies"])
        )

    if role in ("overall", "guesser") and stats["guess_accuracies"]:
        entry.avg_guess_accuracy = (
            sum(stats["guess_accuracies"]) / len(stats["guess_accuracies"])
        )

    return entry


def build_leaderboard(results: list[BenchmarkResult]) -> Leaderboard:
    """
    Build a complete leaderboard from benchmark results.

    Args:
        results: List of benchmark results

    Returns:
        Leaderboard with all views
    """
    # Extract all unique model IDs
    model_ids = set()
    for result in results:
        if result.error:
            continue
        model_ids.update(result.red_models.values())
        model_ids.update(result.blue_models.values())

    # Build overall leaderboard
    overall_entries = []
    for model_id in model_ids:
        stats = _extract_model_stats(results, model_id, "overall")
        entry = _create_leaderboard_entry(model_id, stats, "overall")
        if entry.games > 0:
            overall_entries.append(entry)

    # Sort by win rate (descending), then by coordination score
    overall_entries.sort(
        key=lambda e: (e.win_rate, e.avg_coordination_score),
        reverse=True,
    )

    # Build cluer leaderboard
    cluer_entries = []
    for model_id in model_ids:
        stats = _extract_model_stats(results, model_id, "cluer")
        entry = _create_leaderboard_entry(model_id, stats, "cluer")
        if entry.games > 0:
            cluer_entries.append(entry)

    cluer_entries.sort(
        key=lambda e: (e.win_rate, e.avg_clue_efficiency or 0),
        reverse=True,
    )

    # Build guesser leaderboard
    guesser_entries = []
    for model_id in model_ids:
        stats = _extract_model_stats(results, model_id, "guesser")
        entry = _create_leaderboard_entry(model_id, stats, "guesser")
        if entry.games > 0:
            guesser_entries.append(entry)

    guesser_entries.sort(
        key=lambda e: (e.win_rate, e.avg_guess_accuracy or 0),
        reverse=True,
    )

    # Build leaderboard by mode
    modes = set(r.mode for r in results if not r.error)
    by_mode = {}
    for mode in modes:
        mode_results = [r for r in results if r.mode == mode and not r.error]
        mode_entries = []
        for model_id in model_ids:
            stats = _extract_model_stats(mode_results, model_id, "overall")
            entry = _create_leaderboard_entry(model_id, stats, "overall")
            if entry.games > 0:
                mode_entries.append(entry)
        mode_entries.sort(
            key=lambda e: (e.win_rate, e.avg_coordination_score),
            reverse=True,
        )
        by_mode[mode.value] = mode_entries

    # Build head-to-head comparisons
    head_to_head = []
    model_list = list(model_ids)
    for i, model_a in enumerate(model_list):
        for model_b in model_list[i + 1:]:
            h2h = _compute_head_to_head(results, model_a, model_b)
            if h2h.games > 0:
                head_to_head.append(h2h)

    # Sort by most decisive matchups
    head_to_head.sort(
        key=lambda h: abs(h.model_a_win_rate - 0.5),
        reverse=True,
    )

    return Leaderboard(
        overall=overall_entries,
        by_cluer=cluer_entries,
        by_guesser=guesser_entries,
        by_mode=by_mode,
        head_to_head=head_to_head,
    )


def _compute_head_to_head(
    results: list[BenchmarkResult],
    model_a: str,
    model_b: str,
) -> HeadToHeadEntry:
    """Compute head-to-head statistics between two models."""
    games = 0
    a_wins = 0
    b_wins = 0
    draws = 0

    for result in results:
        if result.error or result.metrics is None:
            continue

        # Check if this is an A vs B game (in homogeneous matchups)
        a_is_red = all(v == model_a for v in result.red_models.values())
        b_is_red = all(v == model_b for v in result.red_models.values())
        a_is_blue = all(v == model_a for v in result.blue_models.values())
        b_is_blue = all(v == model_b for v in result.blue_models.values())

        if (a_is_red and b_is_blue) or (b_is_red and a_is_blue):
            games += 1
            if result.winner is None:
                draws += 1
            elif (a_is_red and result.winner == Team.RED) or \
                 (a_is_blue and result.winner == Team.BLUE):
                a_wins += 1
            else:
                b_wins += 1

    if games == 0:
        return HeadToHeadEntry(
            model_a=model_a,
            model_b=model_b,
            games=0,
            model_a_wins=0,
            model_b_wins=0,
            draws=0,
            model_a_win_rate=0.5,
            model_a_win_rate_ci=ConfidenceInterval(lower=0.0, upper=1.0),
        )

    a_win_rate = a_wins / games if games > 0 else 0.5
    ci = wilson_score_interval(a_wins, games)

    return HeadToHeadEntry(
        model_a=model_a,
        model_b=model_b,
        games=games,
        model_a_wins=a_wins,
        model_b_wins=b_wins,
        draws=draws,
        model_a_win_rate=a_win_rate,
        model_a_win_rate_ci=ci,
    )


def export_leaderboard_markdown(leaderboard: Leaderboard) -> str:
    """Export leaderboard to Markdown format."""
    lines = ["# Benchmark Leaderboard", ""]

    # Overall
    lines.append("## Overall Rankings")
    lines.append("")
    lines.append("| Rank | Model | Games | Win Rate | 95% CI | Coord. Score | ToM |")
    lines.append("|------|-------|-------|----------|--------|--------------|-----|")

    for i, entry in enumerate(leaderboard.overall, 1):
        ci = f"[{entry.win_rate_ci.lower:.2f}, {entry.win_rate_ci.upper:.2f}]"
        lines.append(
            f"| {i} | {entry.model} | {entry.games} | "
            f"{entry.win_rate:.1%} | {ci} | "
            f"{entry.avg_coordination_score:.3f} | {entry.avg_theory_of_mind:.3f} |"
        )

    lines.append("")

    # By Cluer
    if leaderboard.by_cluer:
        lines.append("## Best Cluers")
        lines.append("")
        lines.append("| Rank | Model | Games | Win Rate | Clue Efficiency |")
        lines.append("|------|-------|-------|----------|-----------------|")

        for i, entry in enumerate(leaderboard.by_cluer[:10], 1):
            eff = f"{entry.avg_clue_efficiency:.3f}" if entry.avg_clue_efficiency else "N/A"
            lines.append(
                f"| {i} | {entry.model} | {entry.games} | "
                f"{entry.win_rate:.1%} | {eff} |"
            )

        lines.append("")

    # By Guesser
    if leaderboard.by_guesser:
        lines.append("## Best Guessers")
        lines.append("")
        lines.append("| Rank | Model | Games | Win Rate | Guess Accuracy |")
        lines.append("|------|-------|-------|----------|----------------|")

        for i, entry in enumerate(leaderboard.by_guesser[:10], 1):
            acc = f"{entry.avg_guess_accuracy:.3f}" if entry.avg_guess_accuracy else "N/A"
            lines.append(
                f"| {i} | {entry.model} | {entry.games} | "
                f"{entry.win_rate:.1%} | {acc} |"
            )

        lines.append("")

    # Head-to-head
    if leaderboard.head_to_head:
        lines.append("## Head-to-Head")
        lines.append("")
        lines.append("| Model A | Model B | Games | A Wins | B Wins | A Win Rate |")
        lines.append("|---------|---------|-------|--------|--------|------------|")

        for h2h in leaderboard.head_to_head[:20]:
            lines.append(
                f"| {h2h.model_a} | {h2h.model_b} | {h2h.games} | "
                f"{h2h.model_a_wins} | {h2h.model_b_wins} | {h2h.model_a_win_rate:.1%} |"
            )

        lines.append("")

    return "\n".join(lines)
