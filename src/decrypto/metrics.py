from __future__ import annotations

import math
from typing import Any

from .models import DecryptoEpisodeRecord, RoundLog, TeamKey


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


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

            # State-conditioned decode accuracy
            bucket = _bucket(team, r)
            if bucket not in by_state:
                by_state[bucket] = {"red": [], "blue": []}
            if decode_action is not None:
                by_state[bucket][team].append(1.0 if decode_action.correct else 0.0)

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
        },
        "adaptation": {
            "decode_success_by_state": {
                bucket: {t: _safe_mean(by_state[bucket][t]) for t in ("red", "blue")}
                for bucket in sorted(by_state.keys())
            },
        },
    }
