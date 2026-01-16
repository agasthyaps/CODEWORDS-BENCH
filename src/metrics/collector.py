"""Metrics collection and computation (M5)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.engine import Team, CardType, Clue, Guess, Pass, DiscussionMessage

from .models import TeamMetrics, EpisodeMetrics, AggregateMetrics

if TYPE_CHECKING:
    from src.runner import ExtendedEpisodeRecord


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _spearman_rank_correlation(x_ranks: list[int], y_ranks: list[int]) -> float | None:
    """
    Spearman correlation for rank lists (no ties assumed).
    Returns None if fewer than 2 points.
    """
    n = len(x_ranks)
    if n < 2:
        return None
    # Spearman rho = 1 - 6 * sum(d^2) / (n*(n^2-1))
    d2 = 0
    for xr, yr in zip(x_ranks, y_ranks):
        d = xr - yr
        d2 += d * d
    denom = n * (n * n - 1)
    if denom == 0:
        return None
    return 1.0 - (6.0 * d2) / denom


def _compute_prediction_tom_for_team(
    episode: "ExtendedEpisodeRecord",
    team: Team,
) -> tuple[
    float | None,  # overlap@k mean
    float | None,  # translated overlap@k mean
    float | None,  # rank correlation mean
    float | None,  # confusion calibration mean
    float | None,  # format compliance rate
    float | None,  # board-word compliance rate (top-k)
    float | None,  # non-board rate in top-k
    float | None,  # cluer confidence mean
    float | None,  # cluer overconfidence rate
    int,  # n_predictions seen (including parse failures)
]:
    """
    Compute prediction-based ToM submetrics for a team.

    Returns:
        (prediction_accuracy_mean, rank_correlation_mean, confusion_calibration_mean, n_predictions)
    """
    # Turn traces are Pydantic models; after v1.1 they may include prediction_trace.
    turn_traces = getattr(episode, "turn_traces", []) or []
    transcript = episode.public_transcript

    overlap_vals: list[float] = []
    translated_overlap_vals: list[float] = []
    rank_vals: list[float] = []
    confusion_vals: list[float] = []
    format_ok = 0
    boardword_ok = 0
    non_board_fracs: list[float] = []
    conf_vals: list[float] = []
    overconf_flags: list[int] = []
    n_predictions = 0

    board_words = set([w.upper() for w in episode.board.words])

    for t in turn_traces:
        if getattr(t, "team", None) != team:
            continue

        pred_trace = getattr(t, "prediction_trace", None)
        if pred_trace is None:
            continue

        n_predictions += 1
        if not pred_trace.parsed_result:
            continue

        predicted_guesses = pred_trace.parsed_result.get("predicted_guesses")
        translated_guesses = (
            pred_trace.parsed_result.get("translated_guesses")
            if "translated_guesses" in pred_trace.parsed_result
            else None
        )
        confusion_risks = pred_trace.parsed_result.get("confusion_risks", [])
        confidence = pred_trace.parsed_result.get("confidence")
        if not isinstance(predicted_guesses, list):
            continue
        format_ok += 1

        if isinstance(confidence, int) and 1 <= confidence <= 5:
            conf_vals.append(float(confidence))

        turn_number = getattr(t, "turn_number", None)
        if turn_number is None:
            continue

        # Actual guesses (order as in transcript)
        actual_guesses: list[str] = []
        wrong_guesses: list[str] = []
        for event in transcript:
            if event.get("event_type") != "guess":
                continue
            if event.get("team") != team.value:
                continue
            if event.get("turn_number") != turn_number:
                continue
            word = event.get("word")
            if isinstance(word, str):
                actual_guesses.append(word.upper())
                if not event.get("correct", False):
                    wrong_guesses.append(word.upper())

        # If team made no guesses, treat as a vacuous prediction target.
        if len(actual_guesses) == 0:
            k = 0
            topk = []
            overlap = 1.0 if len(predicted_guesses) == 0 else 0.0
            overlap_vals.append(overlap)
            if isinstance(translated_guesses, list):
                translated_overlap_vals.append(1.0 if len(translated_guesses) == 0 else 0.0)
            boardword_ok += 1  # vacuously true
            non_board_fracs.append(0.0)
            if isinstance(confidence, int) and confidence >= 4 and overlap < 0.5:
                overconf_flags.append(1)
            else:
                overconf_flags.append(0)
            continue

        actual_set = set(actual_guesses)

        # overlap@k (recall@k) where k = number of actual guesses this turn
        k = len(actual_guesses)
        pred_norm = [w.upper() for w in predicted_guesses if isinstance(w, str)]
        topk = pred_norm[:k]
        overlap = len(set(topk).intersection(actual_set)) / len(actual_set)
        overlap_vals.append(overlap)

        # Translated overlap@k (semantic mapping to board words)
        if isinstance(translated_guesses, list):
            trans_norm = [w.upper() for w in translated_guesses if isinstance(w, str)]
            trans_topk = trans_norm[:k]
            trans_overlap = len(set(trans_topk).intersection(actual_set)) / len(actual_set)
            translated_overlap_vals.append(trans_overlap)

        # Compliance: board words only in top-k (separate from ToM)
        if k == 0:
            boardword_ok += 1
            non_board_fracs.append(0.0)
        else:
            non_board = [w for w in topk if w not in board_words]
            non_board_frac = len(non_board) / k
            non_board_fracs.append(non_board_frac)
            if len(non_board) == 0:
                boardword_ok += 1

        # Overconfidence: high confidence but low overlap@k (thresholds can be tuned later)
        if isinstance(confidence, int) and confidence >= 4 and overlap < 0.5:
            overconf_flags.append(1)
        else:
            overconf_flags.append(0)

        # Rank correlation on intersection items
        predicted_set = set(pred_norm)
        common = [w for w in actual_guesses if w in predicted_set]
        if len(common) >= 2:
            pred_index = {w.upper(): i for i, w in enumerate(pred_norm)}
            x_ranks = [pred_index[w] for w in common if w in pred_index]
            y_ranks = [actual_guesses.index(w) for w in common if w in pred_index]
            rho = _spearman_rank_correlation(x_ranks, y_ranks)
            if rho is not None:
                rank_vals.append(rho)

        # Confusion calibration
        if wrong_guesses:
            risk_words = set()
            if isinstance(confusion_risks, list):
                for r in confusion_risks:
                    if isinstance(r, dict):
                        w = r.get("word")
                        if isinstance(w, str):
                            risk_words.add(w.upper())
                    elif isinstance(r, str):
                        risk_words.add(r.upper())
            conf = len(risk_words.intersection(set(wrong_guesses))) / len(wrong_guesses)
            confusion_vals.append(conf)

    format_rate = (format_ok / n_predictions) if n_predictions > 0 else None
    boardword_rate = (boardword_ok / format_ok) if format_ok > 0 else None
    non_board_rate = _safe_mean(non_board_fracs)
    conf_mean = _safe_mean([float(x) for x in conf_vals])
    overconf_rate = (_safe_mean([float(x) for x in overconf_flags]) if overconf_flags else None)

    return (
        _safe_mean(overlap_vals),
        _safe_mean(translated_overlap_vals),
        _safe_mean(rank_vals),
        _safe_mean(confusion_vals),
        format_rate,
        boardword_rate,
        non_board_rate,
        conf_mean,
        overconf_rate,
        n_predictions,
    )


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _point_biserial_corr(xs: list[float], ys01: list[float]) -> float | None:
    """
    Point-biserial correlation between continuous xs and binary ys in {0,1}.
    Equivalent to Pearson correlation when ys is coded 0/1.
    """
    return _pearson_corr(xs, ys01)


def _spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    """
    Spearman correlation via ranking (average ranks for ties).
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        # average rank for ties
        indexed = list(enumerate(vals))
        indexed.sort(key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0  # ranks are 1-based
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson_corr(rx, ry)


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

    # Theory of Mind score (v1.1+): overlap@k (mind-modeling), NOT format compliance.
    (
        overlap_at_k,
        translated_overlap_at_k,
        pred_rank,
        pred_conf,
        fmt_rate,
        boardword_rate,
        non_board_rate,
        cluer_conf_mean,
        cluer_overconf_rate,
        n_pred,
    ) = _compute_prediction_tom_for_team(episode, team)
    # Default ToM score: translated overlap if available, else strict overlap, else fallback.
    theory_of_mind_score = (
        translated_overlap_at_k
        if translated_overlap_at_k is not None
        else (overlap_at_k if overlap_at_k is not None else clue_efficiency)
    )

    # Guesser calibration: confidence + correctness (per-turn)
    guess_conf: list[float] = []
    guess_success: list[float] = []
    guess_overconf_flags: list[float] = []

    # Build quick lookup: for each turn, was there any wrong guess?
    wrong_by_turn: dict[int, bool] = {}
    for event in transcript:
        if event.get("event_type") != "guess":
            continue
        if event.get("team") != team.value:
            continue
        tn = event.get("turn_number")
        if not isinstance(tn, int):
            continue
        if not event.get("correct", False):
            wrong_by_turn[tn] = True

    for tt in team_turns:
        gt = getattr(tt, "guess_trace", None)
        if gt is None or not gt.parsed_result:
            continue
        conf = gt.parsed_result.get("confidence")
        if not (isinstance(conf, int) and 1 <= conf <= 5):
            continue
        tn = getattr(tt, "turn_number", None)
        if not isinstance(tn, int):
            continue
        had_wrong = wrong_by_turn.get(tn, False)
        success = 0.0 if had_wrong else 1.0
        guess_conf.append(float(conf))
        guess_success.append(success)
        guess_overconf_flags.append(1.0 if (conf >= 4 and had_wrong) else 0.0)

    guess_conf_mean = _safe_mean(guess_conf)
    guess_overconf_rate = _safe_mean(guess_overconf_flags)
    guess_n = len(guess_conf)
    guess_pb = _point_biserial_corr(guess_conf, guess_success) if guess_n >= 2 else None
    guess_spear = _spearman_corr(guess_conf, guess_success) if guess_n >= 2 else None

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
        theory_of_mind_score=theory_of_mind_score,
        tom_predictions_count=n_pred,
        tom_overlap_at_k=overlap_at_k,
        tom_translated_overlap_at_k=translated_overlap_at_k,
        tom_rank_correlation=pred_rank,
        tom_confusion_calibration=pred_conf,
        tom_format_compliance_rate=fmt_rate,
        tom_boardword_compliance_rate=boardword_rate,
        tom_non_board_rate_top_k=non_board_rate,
        cluer_confidence_mean=cluer_conf_mean,
        cluer_overconfidence_rate=cluer_overconf_rate,
        guesser_confidence_mean=guess_conf_mean,
        guesser_overconfidence_rate=guess_overconf_rate,
        guesser_confidence_correctness_n=guess_n,
        guesser_confidence_correctness_point_biserial=guess_pb,
        guesser_confidence_correctness_spearman=guess_spear,
    )


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
            avg_theory_of_mind_red=0.0,
            avg_theory_of_mind_blue=0.0,
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

    # Theory of mind
    tom_red = [m.red_metrics.theory_of_mind_score for m in episode_metrics]
    tom_blue = [m.blue_metrics.theory_of_mind_score for m in episode_metrics]
    avg_tom_red = sum(tom_red) / n
    avg_tom_blue = sum(tom_blue) / n

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
        avg_theory_of_mind_red=avg_tom_red,
        avg_theory_of_mind_blue=avg_tom_blue,
        assassin_rate=assassin_rate,
        avg_clue_efficiency_red=avg_clue_eff_red,
        avg_clue_efficiency_blue=avg_clue_eff_blue,
        avg_guess_accuracy_red=avg_guess_acc_red,
        avg_guess_accuracy_blue=avg_guess_acc_blue,
        avg_consensus_rate_red=avg_cons_red,
        avg_consensus_rate_blue=avg_cons_blue,
    )
