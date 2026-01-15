"""Export functions for metrics (M5)."""

from __future__ import annotations

import csv
import io
import json
from typing import Literal

from .models import EpisodeMetrics, AggregateMetrics, TeamMetrics


def export_episode_metrics_json(metrics: EpisodeMetrics) -> str:
    """Export episode metrics to JSON string."""
    return json.dumps(metrics.model_dump(mode="json"), indent=2)


def export_aggregate_metrics_json(metrics: AggregateMetrics) -> str:
    """Export aggregate metrics to JSON string."""
    return json.dumps(metrics.model_dump(mode="json"), indent=2)


def export_episodes_csv(metrics_list: list[EpisodeMetrics]) -> str:
    """
    Export multiple episode metrics to CSV string.

    Each row represents one episode with flattened team metrics.
    """
    if not metrics_list:
        return ""

    output = io.StringIO(newline='')
    writer = csv.writer(output)

    # Headers
    headers = [
        "episode_id",
        "winner",
        "turns_to_win",
        # Red team
        "red_words_cleared",
        "red_assassin_hit",
        "red_total_clues",
        "red_avg_clue_number",
        "red_clue_efficiency",
        "red_total_guesses",
        "red_correct_guesses",
        "red_wrong_guesses",
        "red_guess_accuracy",
        "red_avg_discussion_rounds",
        "red_consensus_rate",
        "red_theory_of_mind",
        "red_tom_overlap_at_k",
        "red_tom_translated_overlap_at_k",
        "red_tom_format_compliance_rate",
        "red_tom_boardword_compliance_rate",
        "red_tom_non_board_rate_top_k",
        "red_cluer_confidence_mean",
        "red_cluer_overconfidence_rate",
        "red_guesser_confidence_mean",
        "red_guesser_overconfidence_rate",
        "red_guesser_confidence_correctness_n",
        "red_guesser_confidence_correctness_point_biserial",
        "red_guesser_confidence_correctness_spearman",
        "red_coordination_score",
        # Blue team
        "blue_words_cleared",
        "blue_assassin_hit",
        "blue_total_clues",
        "blue_avg_clue_number",
        "blue_clue_efficiency",
        "blue_total_guesses",
        "blue_correct_guesses",
        "blue_wrong_guesses",
        "blue_guess_accuracy",
        "blue_avg_discussion_rounds",
        "blue_consensus_rate",
        "blue_theory_of_mind",
        "blue_tom_overlap_at_k",
        "blue_tom_translated_overlap_at_k",
        "blue_tom_format_compliance_rate",
        "blue_tom_boardword_compliance_rate",
        "blue_tom_non_board_rate_top_k",
        "blue_cluer_confidence_mean",
        "blue_cluer_overconfidence_rate",
        "blue_guesser_confidence_mean",
        "blue_guesser_overconfidence_rate",
        "blue_guesser_confidence_correctness_n",
        "blue_guesser_confidence_correctness_point_biserial",
        "blue_guesser_confidence_correctness_spearman",
        "blue_coordination_score",
    ]
    writer.writerow(headers)

    # Data rows
    for m in metrics_list:
        row = [
            m.episode_id,
            m.winner.value if m.winner else "DRAW",
            m.turns_to_win,
            # Red
            m.red_metrics.words_cleared,
            m.red_metrics.assassin_hit,
            m.red_metrics.total_clues,
            f"{m.red_metrics.avg_clue_number:.2f}",
            f"{m.red_metrics.clue_efficiency:.3f}",
            m.red_metrics.total_guesses,
            m.red_metrics.correct_guesses,
            m.red_metrics.wrong_guesses,
            f"{m.red_metrics.guess_accuracy:.3f}",
            f"{m.red_metrics.avg_discussion_rounds:.2f}",
            f"{m.red_metrics.consensus_rate:.3f}",
            f"{m.red_metrics.theory_of_mind_score:.3f}",
            "" if m.red_metrics.tom_overlap_at_k is None else f"{m.red_metrics.tom_overlap_at_k:.3f}",
            "" if m.red_metrics.tom_translated_overlap_at_k is None else f"{m.red_metrics.tom_translated_overlap_at_k:.3f}",
            "" if m.red_metrics.tom_format_compliance_rate is None else f"{m.red_metrics.tom_format_compliance_rate:.3f}",
            "" if m.red_metrics.tom_boardword_compliance_rate is None else f"{m.red_metrics.tom_boardword_compliance_rate:.3f}",
            "" if m.red_metrics.tom_non_board_rate_top_k is None else f"{m.red_metrics.tom_non_board_rate_top_k:.3f}",
            "" if m.red_metrics.cluer_confidence_mean is None else f"{m.red_metrics.cluer_confidence_mean:.3f}",
            "" if m.red_metrics.cluer_overconfidence_rate is None else f"{m.red_metrics.cluer_overconfidence_rate:.3f}",
            "" if m.red_metrics.guesser_confidence_mean is None else f"{m.red_metrics.guesser_confidence_mean:.3f}",
            "" if m.red_metrics.guesser_overconfidence_rate is None else f"{m.red_metrics.guesser_overconfidence_rate:.3f}",
            m.red_metrics.guesser_confidence_correctness_n,
            "" if m.red_metrics.guesser_confidence_correctness_point_biserial is None else f"{m.red_metrics.guesser_confidence_correctness_point_biserial:.3f}",
            "" if m.red_metrics.guesser_confidence_correctness_spearman is None else f"{m.red_metrics.guesser_confidence_correctness_spearman:.3f}",
            f"{m.red_coordination_score:.3f}",
            # Blue
            m.blue_metrics.words_cleared,
            m.blue_metrics.assassin_hit,
            m.blue_metrics.total_clues,
            f"{m.blue_metrics.avg_clue_number:.2f}",
            f"{m.blue_metrics.clue_efficiency:.3f}",
            m.blue_metrics.total_guesses,
            m.blue_metrics.correct_guesses,
            m.blue_metrics.wrong_guesses,
            f"{m.blue_metrics.guess_accuracy:.3f}",
            f"{m.blue_metrics.avg_discussion_rounds:.2f}",
            f"{m.blue_metrics.consensus_rate:.3f}",
            f"{m.blue_metrics.theory_of_mind_score:.3f}",
            "" if m.blue_metrics.tom_overlap_at_k is None else f"{m.blue_metrics.tom_overlap_at_k:.3f}",
            "" if m.blue_metrics.tom_translated_overlap_at_k is None else f"{m.blue_metrics.tom_translated_overlap_at_k:.3f}",
            "" if m.blue_metrics.tom_format_compliance_rate is None else f"{m.blue_metrics.tom_format_compliance_rate:.3f}",
            "" if m.blue_metrics.tom_boardword_compliance_rate is None else f"{m.blue_metrics.tom_boardword_compliance_rate:.3f}",
            "" if m.blue_metrics.tom_non_board_rate_top_k is None else f"{m.blue_metrics.tom_non_board_rate_top_k:.3f}",
            "" if m.blue_metrics.cluer_confidence_mean is None else f"{m.blue_metrics.cluer_confidence_mean:.3f}",
            "" if m.blue_metrics.cluer_overconfidence_rate is None else f"{m.blue_metrics.cluer_overconfidence_rate:.3f}",
            "" if m.blue_metrics.guesser_confidence_mean is None else f"{m.blue_metrics.guesser_confidence_mean:.3f}",
            "" if m.blue_metrics.guesser_overconfidence_rate is None else f"{m.blue_metrics.guesser_overconfidence_rate:.3f}",
            m.blue_metrics.guesser_confidence_correctness_n,
            "" if m.blue_metrics.guesser_confidence_correctness_point_biserial is None else f"{m.blue_metrics.guesser_confidence_correctness_point_biserial:.3f}",
            "" if m.blue_metrics.guesser_confidence_correctness_spearman is None else f"{m.blue_metrics.guesser_confidence_correctness_spearman:.3f}",
            f"{m.blue_coordination_score:.3f}",
        ]
        writer.writerow(row)

    return output.getvalue()


def export_aggregate_csv(metrics: AggregateMetrics) -> str:
    """Export aggregate metrics to CSV string (single row)."""
    output = io.StringIO(newline='')
    writer = csv.writer(output)

    headers = [
        "episodes",
        "win_rate_red",
        "win_rate_blue",
        "draw_rate",
        "avg_turns_to_win",
        "std_turns_to_win",
        "avg_coordination_red",
        "avg_coordination_blue",
        "avg_tom_red",
        "avg_tom_blue",
        "assassin_rate",
        "avg_clue_efficiency_red",
        "avg_clue_efficiency_blue",
        "avg_guess_accuracy_red",
        "avg_guess_accuracy_blue",
    ]
    writer.writerow(headers)

    row = [
        metrics.episodes,
        f"{metrics.win_rate_red:.3f}",
        f"{metrics.win_rate_blue:.3f}",
        f"{metrics.draw_rate:.3f}",
        f"{metrics.avg_turns_to_win:.2f}",
        f"{metrics.std_turns_to_win:.2f}",
        f"{metrics.avg_coordination_score_red:.3f}",
        f"{metrics.avg_coordination_score_blue:.3f}",
        f"{metrics.avg_theory_of_mind_red:.3f}",
        f"{metrics.avg_theory_of_mind_blue:.3f}",
        f"{metrics.assassin_rate:.3f}",
        f"{metrics.avg_clue_efficiency_red:.3f}",
        f"{metrics.avg_clue_efficiency_blue:.3f}",
        f"{metrics.avg_guess_accuracy_red:.3f}",
        f"{metrics.avg_guess_accuracy_blue:.3f}",
    ]
    writer.writerow(row)

    return output.getvalue()


def export_episode_markdown(metrics: EpisodeMetrics) -> str:
    """Export episode metrics to Markdown table."""
    lines = [
        f"# Episode {metrics.episode_id}",
        "",
        f"**Winner:** {metrics.winner.value if metrics.winner else 'Draw'}",
        f"**Turns:** {metrics.turns_to_win}",
        "",
        "## Team Comparison",
        "",
        "| Metric | Red | Blue |",
        "|--------|-----|------|",
        f"| Words Cleared | {metrics.red_metrics.words_cleared} | {metrics.blue_metrics.words_cleared} |",
        f"| Assassin Hit | {metrics.red_metrics.assassin_hit} | {metrics.blue_metrics.assassin_hit} |",
        f"| Total Clues | {metrics.red_metrics.total_clues} | {metrics.blue_metrics.total_clues} |",
        f"| Avg Clue Number | {metrics.red_metrics.avg_clue_number:.2f} | {metrics.blue_metrics.avg_clue_number:.2f} |",
        f"| Clue Efficiency | {metrics.red_metrics.clue_efficiency:.3f} | {metrics.blue_metrics.clue_efficiency:.3f} |",
        f"| Total Guesses | {metrics.red_metrics.total_guesses} | {metrics.blue_metrics.total_guesses} |",
        f"| Correct Guesses | {metrics.red_metrics.correct_guesses} | {metrics.blue_metrics.correct_guesses} |",
        f"| Guess Accuracy | {metrics.red_metrics.guess_accuracy:.3f} | {metrics.blue_metrics.guess_accuracy:.3f} |",
        f"| Avg Discussion Rounds | {metrics.red_metrics.avg_discussion_rounds:.2f} | {metrics.blue_metrics.avg_discussion_rounds:.2f} |",
        f"| Consensus Rate | {metrics.red_metrics.consensus_rate:.3f} | {metrics.blue_metrics.consensus_rate:.3f} |",
        f"| Theory of Mind | {metrics.red_metrics.theory_of_mind_score:.3f} | {metrics.blue_metrics.theory_of_mind_score:.3f} |",
        f"| ToM Overlap@k | {metrics.red_metrics.tom_overlap_at_k if metrics.red_metrics.tom_overlap_at_k is not None else 'N/A'} | {metrics.blue_metrics.tom_overlap_at_k if metrics.blue_metrics.tom_overlap_at_k is not None else 'N/A'} |",
        f"| ToM Format Compliance | {metrics.red_metrics.tom_format_compliance_rate if metrics.red_metrics.tom_format_compliance_rate is not None else 'N/A'} | {metrics.blue_metrics.tom_format_compliance_rate if metrics.blue_metrics.tom_format_compliance_rate is not None else 'N/A'} |",
        f"| ToM Board-Word Compliance | {metrics.red_metrics.tom_boardword_compliance_rate if metrics.red_metrics.tom_boardword_compliance_rate is not None else 'N/A'} | {metrics.blue_metrics.tom_boardword_compliance_rate if metrics.blue_metrics.tom_boardword_compliance_rate is not None else 'N/A'} |",
        f"| Guesser Confidence Mean | {metrics.red_metrics.guesser_confidence_mean if metrics.red_metrics.guesser_confidence_mean is not None else 'N/A'} | {metrics.blue_metrics.guesser_confidence_mean if metrics.blue_metrics.guesser_confidence_mean is not None else 'N/A'} |",
        f"| Guesser Overconfidence Rate | {metrics.red_metrics.guesser_overconfidence_rate if metrics.red_metrics.guesser_overconfidence_rate is not None else 'N/A'} | {metrics.blue_metrics.guesser_overconfidence_rate if metrics.blue_metrics.guesser_overconfidence_rate is not None else 'N/A'} |",
        f"| **Coordination Score** | **{metrics.red_coordination_score:.3f}** | **{metrics.blue_coordination_score:.3f}** |",
        "",
    ]
    return "\n".join(lines)


def export_aggregate_markdown(metrics: AggregateMetrics) -> str:
    """Export aggregate metrics to Markdown table."""
    lines = [
        "# Aggregate Metrics",
        "",
        f"**Total Episodes:** {metrics.episodes}",
        "",
        "## Win Rates",
        "",
        "| Team | Win Rate |",
        "|------|----------|",
        f"| Red | {metrics.win_rate_red:.1%} |",
        f"| Blue | {metrics.win_rate_blue:.1%} |",
        f"| Draw | {metrics.draw_rate:.1%} |",
        "",
        "## Game Length",
        "",
        f"- **Average Turns to Win:** {metrics.avg_turns_to_win:.2f}",
        f"- **Std Dev:** {metrics.std_turns_to_win:.2f}",
        f"- **Assassin Hit Rate:** {metrics.assassin_rate:.1%}",
        "",
        "## Coordination Metrics",
        "",
        "| Metric | Red | Blue |",
        "|--------|-----|------|",
        f"| Avg Coordination Score | {metrics.avg_coordination_score_red:.3f} | {metrics.avg_coordination_score_blue:.3f} |",
        f"| Avg Theory of Mind | {metrics.avg_theory_of_mind_red:.3f} | {metrics.avg_theory_of_mind_blue:.3f} |",
        f"| Avg Clue Efficiency | {metrics.avg_clue_efficiency_red:.3f} | {metrics.avg_clue_efficiency_blue:.3f} |",
        f"| Avg Guess Accuracy | {metrics.avg_guess_accuracy_red:.3f} | {metrics.avg_guess_accuracy_blue:.3f} |",
        f"| Avg Consensus Rate | {metrics.avg_consensus_rate_red:.3f} | {metrics.avg_consensus_rate_blue:.3f} |",
        "",
    ]
    return "\n".join(lines)


def export_metrics(
    metrics: EpisodeMetrics | AggregateMetrics | list[EpisodeMetrics],
    format: Literal["json", "csv", "markdown"] = "json",
) -> str:
    """
    Export metrics in the specified format.

    Args:
        metrics: Episode metrics, aggregate metrics, or list of episode metrics
        format: Output format ("json", "csv", "markdown")

    Returns:
        Formatted string
    """
    if isinstance(metrics, list):
        # List of episode metrics
        if format == "json":
            return json.dumps(
                [m.model_dump(mode="json") for m in metrics],
                indent=2,
            )
        elif format == "csv":
            return export_episodes_csv(metrics)
        elif format == "markdown":
            return "\n\n---\n\n".join(
                export_episode_markdown(m) for m in metrics
            )

    elif isinstance(metrics, AggregateMetrics):
        if format == "json":
            return export_aggregate_metrics_json(metrics)
        elif format == "csv":
            return export_aggregate_csv(metrics)
        elif format == "markdown":
            return export_aggregate_markdown(metrics)

    elif isinstance(metrics, EpisodeMetrics):
        if format == "json":
            return export_episode_metrics_json(metrics)
        elif format == "csv":
            return export_episodes_csv([metrics])
        elif format == "markdown":
            return export_episode_markdown(metrics)

    raise ValueError(f"Unknown format: {format}")
