"""Metrics collection and computation (M5)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from src.engine import Team, CardType, Clue, Guess, Pass, DiscussionMessage

from .models import TeamMetrics, EpisodeMetrics, AggregateMetrics

if TYPE_CHECKING:
    from src.runner import ExtendedEpisodeRecord
    from src.runner.orchestrator import TurnTraces


def compute_team_metrics(
    episode: "ExtendedEpisodeRecord",
    team: Team,
) -> TeamMetrics:
    """
    Compute metrics for a single team from an episode.

    Args:
        episode: The episode record
        team: The team to compute metrics for

    Returns:
        TeamMetrics for the specified team
    """
    transcript = episode.public_transcript
    turn_traces = episode.turn_traces

    # Filter events by team
    team_clues: list[dict] = []
    team_guesses: list[dict] = []
    team_discussions: list[dict] = []

    for event in transcript:
        event_team = event.get("team")
        if event_team == team.value:
            event_type = event.get("event_type")
            if event_type == "clue":
                team_clues.append(event)
            elif event_type == "guess":
                team_guesses.append(event)
            elif event_type == "discussion":
                team_discussions.append(event)

    # Clue metrics
    total_clues = len(team_clues)
    clue_numbers = [c.get("number", 0) for c in team_clues]
    # Handle unlimited clues (-1) as 0 for averaging
    avg_clue_number = 0.0
    if total_clues > 0:
        valid_numbers = [n for n in clue_numbers if n > 0]
        avg_clue_number = sum(valid_numbers) / len(valid_numbers) if valid_numbers else 0.0

    # Guess metrics
    total_guesses = len(team_guesses)
    correct_guesses = sum(1 for g in team_guesses if g.get("correct", False))
    wrong_guesses = total_guesses - correct_guesses

    # Break down wrong guesses
    team_card_type = CardType.RED if team == Team.RED else CardType.BLUE
    opponent_card_type = CardType.BLUE if team == Team.RED else CardType.RED

    opponent_guesses = sum(
        1 for g in team_guesses
        if g.get("result") == opponent_card_type.value
    )
    neutral_guesses = sum(
        1 for g in team_guesses
        if g.get("result") == CardType.NEUTRAL.value
    )

    guess_accuracy = correct_guesses / total_guesses if total_guesses > 0 else 0.0

    # Clue efficiency: correct_guesses / sum(clue_numbers)
    # For unlimited clues (-1), we use the actual number of correct guesses that turn
    sum_clue_numbers = sum(n for n in clue_numbers if n > 0)
    # Add unlimited clues as 9 (max possible)
    unlimited_clues = sum(1 for n in clue_numbers if n <= 0)
    sum_clue_numbers += unlimited_clues * 9

    clue_efficiency = correct_guesses / sum_clue_numbers if sum_clue_numbers > 0 else 0.0

    # Check for assassin hit
    assassin_hit = any(
        g.get("result") == CardType.ASSASSIN.value
        for g in team_guesses
    )

    # Words cleared (count of correct guesses)
    words_cleared = correct_guesses

    # Discussion metrics
    # Group discussions by turn
    discussions_by_turn: dict[int, list[dict]] = {}
    for d in team_discussions:
        turn = d.get("turn_number", 0)
        if turn not in discussions_by_turn:
            discussions_by_turn[turn] = []
        discussions_by_turn[turn].append(d)

    # Count rounds per turn (2 messages = 1 round)
    rounds_per_turn = []
    for turn_msgs in discussions_by_turn.values():
        rounds = math.ceil(len(turn_msgs) / 2)
        rounds_per_turn.append(rounds)

    avg_discussion_rounds = (
        sum(rounds_per_turn) / len(rounds_per_turn)
        if rounds_per_turn else 0.0
    )

    # Total discussion length
    total_discussion_length = sum(
        len(d.get("content", "")) for d in team_discussions
    )
    avg_discussion_length = (
        total_discussion_length // len(discussions_by_turn)
        if discussions_by_turn else 0
    )

    # Consensus rate: check turn traces for real consensus (both YES + matching TOP lists)
    team_turns = [t for t in turn_traces if t.team == team]
    turns_with_consensus = 0

    def _top_lists_match(list1: list[str] | None, list2: list[str] | None) -> bool:
        """Check if two TOP word lists match (order-independent)."""
        if list1 is None or list2 is None:
            return False
        set1 = set(w.upper() for w in list1)
        set2 = set(w.upper() for w in list2)
        return set1 == set2

    for turn_trace in team_turns:
        # Check if discussion traces indicate real consensus
        if turn_trace.discussion_traces and len(turn_trace.discussion_traces) >= 2:
            # Look for consecutive CONSENSUS: YES with matching TOP lists
            last_traces = turn_trace.discussion_traces[-2:]
            both_said_yes = all(
                trace.parsed_result and trace.parsed_result.get("consensus", False)
                for trace in last_traces
            )
            if both_said_yes:
                # Verify TOP lists match
                top1 = last_traces[0].parsed_result.get("top_words") if last_traces[0].parsed_result else None
                top2 = last_traces[1].parsed_result.get("top_words") if last_traces[1].parsed_result else None
                if _top_lists_match(top1, top2):
                    turns_with_consensus += 1

    consensus_rate = turns_with_consensus / len(team_turns) if team_turns else 0.0

    # ToM metrics: Cluer Surprise and Interpretability
    tom_metrics = compute_tom_metrics(episode, team)

    return TeamMetrics(
        team=team,
        words_cleared=words_cleared,
        assassin_hit=assassin_hit,
        total_clues=total_clues,
        avg_clue_number=avg_clue_number,
        clue_efficiency=clue_efficiency,
        total_guesses=total_guesses,
        correct_guesses=correct_guesses,
        wrong_guesses=wrong_guesses,
        opponent_guesses=opponent_guesses,
        neutral_guesses=neutral_guesses,
        guess_accuracy=guess_accuracy,
        avg_discussion_rounds=avg_discussion_rounds,
        consensus_rate=consensus_rate,
        avg_discussion_length=avg_discussion_length,
        avg_cluer_surprise=tom_metrics.get("avg_surprise"),
        avg_clue_interpretability=tom_metrics.get("avg_interpretability"),
        top1_match_rate=tom_metrics.get("top1_match_rate"),
    )


def compute_cluer_surprise(
    predicted_success: float | None,
    actual_guesses: list[dict],
    predicted_targets: list[str] | None,
) -> float | None:
    """
    Compute Brier score for cluer's prediction accuracy.
    
    Brier score = (predicted_success - actual_outcome)^2
    Lower is better (0 = perfect calibration).
    
    Args:
        predicted_success: Cluer's predicted probability of success (0.0-1.0)
        actual_guesses: List of guess events from the turn
        predicted_targets: List of words the cluer expected to be guessed
    
    Returns:
        Brier score or None if no prediction available
    """
    if predicted_success is None:
        return None
    
    # Compute actual outcome: 1.0 if all predicted targets were guessed correctly
    if not actual_guesses:
        actual_outcome = 0.0
    elif predicted_targets:
        # Check if all predicted targets were guessed correctly
        correct_guesses = {g.get("word", "").upper() for g in actual_guesses if g.get("correct", False)}
        predicted_set = {t.upper() for t in predicted_targets}
        actual_outcome = 1.0 if predicted_set.issubset(correct_guesses) else 0.0
    else:
        # No predicted targets - check if any correct guesses
        actual_outcome = 1.0 if any(g.get("correct", False) for g in actual_guesses) else 0.0
    
    return (predicted_success - actual_outcome) ** 2


def compute_clue_interpretability(
    predicted_targets: list[str] | None,
    actual_guesses: list[dict],
) -> dict[str, Any]:
    """
    Compute how well guessers interpreted the cluer's intended meaning.
    
    Args:
        predicted_targets: Cluer's intended target words (in order)
        actual_guesses: List of guess events from the turn
    
    Returns:
        dict with:
            - jaccard: Set overlap / union (0.0-1.0)
            - top1_match: Whether first guess matched first target
    """
    if not predicted_targets or not actual_guesses:
        return {"jaccard": None, "top1_match": None}
    
    # Get actual guessed words (in order)
    guessed_words = [g.get("word", "").upper() for g in actual_guesses]
    predicted_upper = [t.upper() for t in predicted_targets]
    
    # Jaccard similarity (set-based)
    predicted_set = set(predicted_upper)
    guessed_set = set(guessed_words)
    
    intersection = len(predicted_set & guessed_set)
    union = len(predicted_set | guessed_set)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Top-1 match (order-sensitive)
    top1_match = (
        guessed_words[0] == predicted_upper[0]
        if guessed_words and predicted_upper
        else False
    )
    
    return {"jaccard": jaccard, "top1_match": top1_match}


def compute_tom_metrics(
    episode: "ExtendedEpisodeRecord",
    team: Team,
) -> dict[str, float | None]:
    """
    Compute Theory of Mind metrics for a team.
    
    Args:
        episode: The episode record
        team: The team to compute metrics for
    
    Returns:
        dict with avg_surprise, avg_interpretability, top1_match_rate
    """
    turn_traces = episode.turn_traces
    transcript = episode.public_transcript
    
    # Filter to team's turns
    team_turns = [t for t in turn_traces if t.team == team]
    
    surprises: list[float] = []
    interpretabilities: list[float] = []
    top1_matches: list[bool] = []
    
    for turn_trace in team_turns:
        turn_num = turn_trace.turn_number
        clue_trace = turn_trace.clue_trace
        
        # Get cluer's predictions
        predicted_success = getattr(clue_trace, 'predicted_success', None)
        predicted_targets = getattr(clue_trace, 'predicted_targets', None)
        
        # Get actual guesses for this turn
        turn_guesses = [
            e for e in transcript
            if e.get("turn_number") == turn_num
            and e.get("team") == team.value
            and e.get("event_type") == "guess"
        ]
        
        # Compute surprise (Brier score)
        surprise = compute_cluer_surprise(predicted_success, turn_guesses, predicted_targets)
        if surprise is not None:
            surprises.append(surprise)
        
        # Compute interpretability
        interp = compute_clue_interpretability(predicted_targets, turn_guesses)
        if interp["jaccard"] is not None:
            interpretabilities.append(interp["jaccard"])
        if interp["top1_match"] is not None:
            top1_matches.append(interp["top1_match"])
    
    return {
        "avg_surprise": sum(surprises) / len(surprises) if surprises else None,
        "avg_interpretability": sum(interpretabilities) / len(interpretabilities) if interpretabilities else None,
        "top1_match_rate": sum(top1_matches) / len(top1_matches) if top1_matches else None,
    }


def compute_coordination_score(metrics: TeamMetrics) -> float:
    """
    Compute the coordination score for a team.

    Formula:
        coordination_score = (
            0.4 * clue_efficiency +
            0.3 * guess_accuracy +
            0.2 * consensus_rate +
            0.1 * (1 / avg_discussion_rounds)  # Capped at 1.0
        )

    Args:
        metrics: Team metrics

    Returns:
        Coordination score in [0, 1] range (approximately)
    """
    # Cap the discussion speed component at 1.0
    discussion_speed = (
        min(1.0, 1.0 / metrics.avg_discussion_rounds)
        if metrics.avg_discussion_rounds > 0
        else 1.0  # Perfect score if no discussion needed
    )

    score = (
        0.4 * metrics.clue_efficiency +
        0.3 * metrics.guess_accuracy +
        0.2 * metrics.consensus_rate +
        0.1 * discussion_speed
    )

    return score


def compute_episode_metrics(episode: "ExtendedEpisodeRecord") -> EpisodeMetrics:
    """
    Compute all metrics for an episode.

    Args:
        episode: The episode record

    Returns:
        EpisodeMetrics with all computed values
    """
    red_metrics = compute_team_metrics(episode, Team.RED)
    blue_metrics = compute_team_metrics(episode, Team.BLUE)

    red_coordination = compute_coordination_score(red_metrics)
    blue_coordination = compute_coordination_score(blue_metrics)

    return EpisodeMetrics(
        episode_id=episode.episode_id,
        winner=episode.winner,
        turns_to_win=episode.total_turns,
        red_metrics=red_metrics,
        blue_metrics=blue_metrics,
        red_coordination_score=red_coordination,
        blue_coordination_score=blue_coordination,
    )


def compute_aggregate_metrics(
    episodes: list["ExtendedEpisodeRecord"],
    config: "GameConfig | None" = None,
) -> AggregateMetrics:
    """
    Compute aggregate metrics across multiple episodes.

    Args:
        episodes: List of episode records
        config: Optional game config (for reference)

    Returns:
        AggregateMetrics with aggregated values
    """
    if not episodes:
        return AggregateMetrics(
            config=config,
            episodes=0,
            win_rate_red=0.0,
            win_rate_blue=0.0,
            draw_rate=0.0,
            avg_turns_to_win=0.0,
            std_turns_to_win=0.0,
            avg_coordination_score_red=0.0,
            avg_coordination_score_blue=0.0,
            assassin_rate=0.0,
        )

    # Compute per-episode metrics
    episode_metrics = [compute_episode_metrics(ep) for ep in episodes]
    n = len(episodes)

    # Win rates
    red_wins = sum(1 for m in episode_metrics if m.winner == Team.RED)
    blue_wins = sum(1 for m in episode_metrics if m.winner == Team.BLUE)
    draws = sum(1 for m in episode_metrics if m.winner is None)

    win_rate_red = red_wins / n
    win_rate_blue = blue_wins / n
    draw_rate = draws / n

    # Turns to win (only for games with a winner)
    games_with_winner = [m for m in episode_metrics if m.winner is not None]
    if games_with_winner:
        turns = [m.turns_to_win for m in games_with_winner]
        avg_turns = sum(turns) / len(turns)
        variance = sum((t - avg_turns) ** 2 for t in turns) / len(turns)
        std_turns = math.sqrt(variance)
    else:
        avg_turns = 0.0
        std_turns = 0.0

    # Coordination scores
    coord_red = [m.red_coordination_score for m in episode_metrics]
    coord_blue = [m.blue_coordination_score for m in episode_metrics]
    avg_coord_red = sum(coord_red) / n
    avg_coord_blue = sum(coord_blue) / n

    # Assassin rate
    assassin_hits = sum(
        1 for m in episode_metrics
        if m.red_metrics.assassin_hit or m.blue_metrics.assassin_hit
    )
    assassin_rate = assassin_hits / n

    # Per-metric averages
    avg_clue_eff_red = sum(m.red_metrics.clue_efficiency for m in episode_metrics) / n
    avg_clue_eff_blue = sum(m.blue_metrics.clue_efficiency for m in episode_metrics) / n
    avg_guess_acc_red = sum(m.red_metrics.guess_accuracy for m in episode_metrics) / n
    avg_guess_acc_blue = sum(m.blue_metrics.guess_accuracy for m in episode_metrics) / n
    avg_cons_red = sum(m.red_metrics.consensus_rate for m in episode_metrics) / n
    avg_cons_blue = sum(m.blue_metrics.consensus_rate for m in episode_metrics) / n

    return AggregateMetrics(
        config=config,
        episodes=n,
        win_rate_red=win_rate_red,
        win_rate_blue=win_rate_blue,
        draw_rate=draw_rate,
        avg_turns_to_win=avg_turns,
        std_turns_to_win=std_turns,
        avg_coordination_score_red=avg_coord_red,
        avg_coordination_score_blue=avg_coord_blue,
        assassin_rate=assassin_rate,
        avg_clue_efficiency_red=avg_clue_eff_red,
        avg_clue_efficiency_blue=avg_clue_eff_blue,
        avg_guess_accuracy_red=avg_guess_acc_red,
        avg_guess_accuracy_blue=avg_guess_acc_blue,
        avg_consensus_rate_red=avg_cons_red,
        avg_consensus_rate_blue=avg_cons_blue,
    )
