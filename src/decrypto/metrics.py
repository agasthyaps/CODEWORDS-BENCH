from __future__ import annotations

import math
from typing import Any

from .models import DecryptoEpisodeRecord, RoundLog, TeamKey


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def code_distance(guess: tuple[int, int, int], actual: tuple[int, int, int]) -> float:
    """
    Compute nuanced distance between two Decrypto codes.
    
    This metric rewards partial correctness:
    - Exact match: 0.0
    - Position-aware: correct digit in correct position = 0 penalty for that position
    - Digit set: having the right digits (wrong positions) gets partial credit
    
    Examples:
        code_distance((1,2,3), (1,2,3)) -> 0.0 (exact match)
        code_distance((1,2,3), (1,4,3)) -> ~0.33 (one position wrong)
        code_distance((1,2,3), (2,3,1)) -> ~1.0 (all positions wrong, but right digits)
        code_distance((1,2,3), (4,4,4)) -> ~1.5 (completely wrong digits)
    
    Formula:
        position_errors = sum(g != a for g,a in zip(guess, actual)) / 3
        digit_set_penalty = 0.5 * (1 - |intersection| / 3)
        return position_errors + digit_set_penalty
    
    Args:
        guess: The guessed code tuple (e.g., (1, 4, 3))
        actual: The actual code tuple (e.g., (1, 2, 3))
    
    Returns:
        Distance score in [0.0, 1.5] range. Lower is better.
    """
    # Position errors: 0, 1, 2, or 3 positions wrong -> normalized to [0, 1]
    position_errors = sum(g != a for g, a in zip(guess, actual)) / 3.0
    
    # Digit set overlap: how many of the guessed digits appear in actual (regardless of position)
    guess_set = set(guess)
    actual_set = set(actual)
    intersection_size = len(guess_set & actual_set)
    
    # Penalty for missing digits: 0 if all 3 match, 0.5 if none match
    digit_set_penalty = 0.5 * (1.0 - intersection_size / 3.0)
    
    return position_errors + digit_set_penalty


def compute_adaptation_rate(episode: DecryptoEpisodeRecord) -> dict[str, Any]:
    """
    Track how intercept quality improves over rounds.
    
    Measures learning rate: as teams observe more clue-code pairs from opponents,
    their intercept guesses should get closer to correct (even if not exact).
    
    Returns:
        - per_round_distance: Average code distance per round (by team)
        - improvement_slope: Linear regression slope (negative = improving)
        - rounds_to_first_intercept: How many rounds until first successful intercept
        - partial_credit_scores: Average distance accounting for near-misses
    """
    # Track intercept distances per round for each team
    distances_by_round: dict[TeamKey, list[tuple[int, float]]] = {"red": [], "blue": []}
    first_intercept_round: dict[TeamKey, int | None] = {"red": None, "blue": None}
    
    for r in episode.rounds:
        round_num = r.round_number
        acts = _team_actions(r)
        
        for team in ("red", "blue"):
            intercept_action = acts.get((team, "intercept"))
            if intercept_action is None:
                continue
                
            # Skip uninformed intercepts (round 1)
            if getattr(intercept_action, "uninformed", False):
                continue
            
            # Get the guess and actual code
            guess = intercept_action.consensus.guess if intercept_action.consensus else None
            
            # Get opponent's actual code for this round from end-of-round reveal.
            # Round actions only contain decode/intercept logs, not clue actions.
            opponent = "blue" if team == "red" else "red"
            actual_code = r.reveal_true_codes.get(opponent)
            if actual_code is None or guess is None:
                continue

            # Ensure both are tuples of 3 ints
            if isinstance(guess, (list, tuple)) and len(guess) == 3:
                guess_tuple = tuple(int(g) for g in guess)
                actual_tuple = tuple(int(a) for a in actual_code)
                
                dist = code_distance(guess_tuple, actual_tuple)
                distances_by_round[team].append((round_num, dist))
                
                # Track first successful intercept
                if intercept_action.correct and first_intercept_round[team] is None:
                    first_intercept_round[team] = round_num
    
    # Compute per-round averages and improvement slope
    result: dict[str, Any] = {
        "per_round_distance": {"red": {}, "blue": {}},
        "improvement_slope": {"red": None, "blue": None},
        "rounds_to_first_intercept": first_intercept_round,
        "partial_credit_scores": {"red": None, "blue": None},
    }
    
    for team in ("red", "blue"):
        data = distances_by_round[team]
        if not data:
            continue
        
        # Group by round
        by_round: dict[int, list[float]] = {}
        for round_num, dist in data:
            if round_num not in by_round:
                by_round[round_num] = []
            by_round[round_num].append(dist)
        
        # Per-round averages
        for round_num, dists in by_round.items():
            result["per_round_distance"][team][round_num] = sum(dists) / len(dists)
        
        # Overall partial credit score (average distance, inverted so higher = better)
        all_distances = [d for _, d in data]
        result["partial_credit_scores"][team] = 1.0 - (sum(all_distances) / len(all_distances) / 1.5)
        
        # Linear regression for improvement slope
        if len(data) >= 2:
            rounds = [r for r, _ in data]
            dists = [d for _, d in data]
            n = len(data)
            
            mean_r = sum(rounds) / n
            mean_d = sum(dists) / n
            
            numerator = sum((r - mean_r) * (d - mean_d) for r, d in zip(rounds, dists))
            denominator = sum((r - mean_r) ** 2 for r in rounds)
            
            if denominator > 0:
                result["improvement_slope"][team] = numerator / denominator
    
    return result


def brier(p: float, y01: float) -> float:
    return (p - y01) ** 2


def log_loss(p: float, y01: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return -(y01 * math.log(p) + (1.0 - y01) * math.log(1.0 - p))


def _team_actions(round_log: RoundLog) -> dict[tuple[TeamKey, str], Any]:
    """Index actions by (team, kind)."""
    out: dict[tuple[TeamKey, str], Any] = {}
    for a in round_log.actions:
        out[(a.team, a.kind)] = a
    return out


def compute_episode_scores(episode: DecryptoEpisodeRecord) -> dict[str, Any]:
    """
    Compute game outcome metrics for a Decrypto episode.
    
    Focuses on observable outcomes rather than introspection:
    - Decode success rates
    - Intercept success rates
    - Confidence calibration (Brier score, log loss)
    - State-conditioned performance
    """
    # Per-team aggregates
    decode_success: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    intercept_success: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    
    # Confidence calibration (using guesser confidence on consensus)
    decode_brier: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    decode_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    intercept_brier: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    intercept_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    decode_bias: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    intercept_bias: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    # Parse success rates
    parse_ok_count: dict[TeamKey, int] = {"red": 0, "blue": 0}
    parse_total: dict[TeamKey, int] = {"red": 0, "blue": 0}

    # State-conditioned buckets (at clue time)
    by_state: dict[str, dict[TeamKey, list[float]]] = {}

    def _bucket(team: TeamKey, round_log: RoundLog) -> str:
        tag = round_log.round_state_at_clue_time[team]
        base = tag.interceptions_state
        if tag.danger:
            return f"{base}_danger"
        return base

    for r in episode.rounds:
        acts = _team_actions(r)

        for team in ("red", "blue"):
            # Decode outcomes
            decode_action = acts.get((team, "decode"))
            if decode_action is not None:
                decode_success[team].append(1.0 if decode_action.correct else 0.0)
                
                # Calibration from consensus confidence
                conf = decode_action.consensus.confidence
                if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
                    p = float(conf)
                    y = 1.0 if decode_action.correct else 0.0
                    decode_brier[team].append(brier(p, y))
                    decode_logloss[team].append(log_loss(p, y))
                    decode_bias[team].append(p - y)
                
                # Parse success tracking
                if decode_action.consensus.parse_ok:
                    parse_ok_count[team] += 1
                parse_total[team] += 1

            # Intercept outcomes
            intercept_action = acts.get((team, "intercept"))
            if intercept_action is not None:
                # Skip uninformed intercepts (round 1)
                if not getattr(intercept_action, "uninformed", False):
                    intercept_success[team].append(1.0 if intercept_action.correct else 0.0)
                    
                    conf = intercept_action.consensus.confidence
                    if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
                        p = float(conf)
                        y = 1.0 if intercept_action.correct else 0.0
                        intercept_brier[team].append(brier(p, y))
                        intercept_logloss[team].append(log_loss(p, y))
                        intercept_bias[team].append(p - y)

            # State-conditioned decode accuracy
            bucket = _bucket(team, r)
            if bucket not in by_state:
                by_state[bucket] = {"red": [], "blue": []}
            if decode_action is not None:
                by_state[bucket][team].append(1.0 if decode_action.correct else 0.0)

    # Compute adaptation rate metrics
    adaptation_metrics = compute_adaptation_rate(episode)
    
    return {
        "outcomes": {
            "decode_success_rate": {t: _safe_mean(decode_success[t]) for t in ("red", "blue")},
            "intercept_success_rate": {t: _safe_mean(intercept_success[t]) for t in ("red", "blue")},
            "parse_success_rate": {
                t: (parse_ok_count[t] / parse_total[t]) if parse_total[t] > 0 else None
                for t in ("red", "blue")
            },
        },
        "calibration": {
            "decode_brier": {t: _safe_mean(decode_brier[t]) for t in ("red", "blue")},
            "decode_log_loss": {t: _safe_mean(decode_logloss[t]) for t in ("red", "blue")},
            "intercept_brier": {t: _safe_mean(intercept_brier[t]) for t in ("red", "blue")},
            "intercept_log_loss": {t: _safe_mean(intercept_logloss[t]) for t in ("red", "blue")},
            "decode_bias": {t: _safe_mean(decode_bias[t]) for t in ("red", "blue")},
            "intercept_bias": {t: _safe_mean(intercept_bias[t]) for t in ("red", "blue")},
        },
        "adaptation": {
            "decode_success_by_state": {
                bucket: {t: _safe_mean(by_state[bucket][t]) for t in ("red", "blue")}
                for bucket in sorted(by_state.keys())
            },
            # Nuanced adaptation metrics
            "intercept_distance_by_round": adaptation_metrics["per_round_distance"],
            "improvement_slope": adaptation_metrics["improvement_slope"],
            "rounds_to_first_intercept": adaptation_metrics["rounds_to_first_intercept"],
            "partial_credit_scores": adaptation_metrics["partial_credit_scores"],
        },
    }
