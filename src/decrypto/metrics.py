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


def _code_key(code: tuple[int, int, int] | None) -> str | None:
    if code is None:
        return None
    return f"{code[0]}-{code[1]}-{code[2]}"


def _normalize_dist(dist: dict[str, float], all_keys: list[str]) -> dict[str, float]:
    """
    Normalize a (possibly partial) distribution onto the 24-code support.
    Missing keys get 0; then we renormalize. If total==0, return uniform.
    """
    cleaned: dict[str, float] = {}
    for k in all_keys:
        v = dist.get(k, 0.0)
        try:
            x = float(v)
        except Exception:
            x = 0.0
        if x < 0:
            x = 0.0
        cleaned[k] = x
    s = sum(cleaned.values())
    if s <= 0:
        u = 1.0 / len(all_keys)
        return {k: u for k in all_keys}
    return {k: v / s for k, v in cleaned.items()}


def _top1_from_dist(dist: dict[str, float]) -> str:
    return max(dist.items(), key=lambda kv: kv[1])[0]


def _team_actions(round_log: RoundLog) -> dict[tuple[TeamKey, str], Any]:
    """
    Index actions by (team, kind).
    """
    out: dict[tuple[TeamKey, str], Any] = {}
    for a in round_log.actions:
        out[(a.team, a.kind)] = a
    return out


def compute_episode_scores(episode: DecryptoEpisodeRecord) -> dict[str, Any]:
    """
    Compute concrete ToM/calibration/adaptation metrics per the tightened spec.
    """
    # Per-team aggregates
    team_tom_hits: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    team_brier: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    team_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    opp_tom_hits: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    leak_brier: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    leak_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    intercept_brier: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    intercept_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    # New ToM: cluer predicts opponent decode + opponent intercept as distributions over 24 codes.
    tom_decode_top1: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    tom_decode_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    tom_intercept_top1: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    tom_intercept_logloss: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    # Calibration events
    overconf_indep: dict[TeamKey, int] = {"red": 0, "blue": 0}
    overconf_cons: dict[TeamKey, int] = {"red": 0, "blue": 0}
    grounding_violations_indep: dict[TeamKey, int] = {"red": 0, "blue": 0}
    grounding_violations_cons: dict[TeamKey, int] = {"red": 0, "blue": 0}
    false_confirmations_indep: dict[TeamKey, int] = {"red": 0, "blue": 0}
    false_confirmations_cons: dict[TeamKey, int] = {"red": 0, "blue": 0}
    hypothesis_refs_indep: dict[TeamKey, int] = {"red": 0, "blue": 0}
    hypothesis_to_confirmed_promotions_within_round: dict[TeamKey, int] = {"red": 0, "blue": 0}
    parse_errors_indep: dict[TeamKey, int] = {"red": 0, "blue": 0}
    parse_errors_cons: dict[TeamKey, int] = {"red": 0, "blue": 0}

    # Evaluator-side mapping states:
    # - hard_confirmed: digit_clue mappings from prior reveals (structural)
    # - soft_supported: digit_theme mappings promoted after correct decode (semantic)
    hard_confirmed_clues: dict[TeamKey, dict[str, set[str]]] = {"red": {str(d): set() for d in range(1, 5)}, "blue": {str(d): set() for d in range(1, 5)}}
    soft_supported_themes: dict[TeamKey, dict[str, set[str]]] = {"red": {str(d): set() for d in range(1, 5)}, "blue": {str(d): set() for d in range(1, 5)}}

    # Rewards: hypothesis maintenance + hedging aligned to soft/hard evidence
    hypothesis_maintained_after_success: dict[TeamKey, list[float]] = {"red": [], "blue": []}
    hedging_alignment: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    # State-conditioned buckets (at clue time)
    by_state: dict[str, dict[TeamKey, list[float]]] = {}

    # Semantic slot reuse (cluer-side, via private slot_themes)
    slot_reuse_buckets: dict[str, dict[TeamKey, list[float]]] = {}
    slot_reuse_overall: dict[TeamKey, list[float]] = {"red": [], "blue": []}

    def _bucket(team: TeamKey, round_log: RoundLog) -> str:
        tag = round_log.round_state_at_clue_time[team]
        base = tag.interceptions_state
        if tag.danger:
            return f"{base}_danger"
        return base

    for idx, r in enumerate(episode.rounds):
        acts = _team_actions(r)

        # Decode outcomes
        red_decode = acts.get(("red", "decode"))
        blue_decode = acts.get(("blue", "decode"))
        if red_decode is None or blue_decode is None:
            continue

        # Intercept outcomes
        red_intercept = acts.get(("red", "intercept"))
        blue_intercept = acts.get(("blue", "intercept"))
        if red_intercept is None or blue_intercept is None:
            continue

        # --- Team-ToM + calibration: from private cluer annotations ---
        for team in ("red", "blue"):
            priv = (r.private or {}).get(team, {})
            ann = priv.get("cluer_annotations") if isinstance(priv, dict) else None
            if not isinstance(ann, dict):
                continue

            # Predicted team guess
            ptg = ann.get("predicted_team_guess")
            if isinstance(ptg, list) and len(ptg) == 3 and all(isinstance(x, (int, float)) for x in ptg):
                ptg_t = tuple(int(x) for x in ptg)
                actual = acts[(team, "decode")].consensus.guess
                if actual is not None:
                    team_tom_hits[team].append(1.0 if ptg_t == actual else 0.0)

            # p_team_correct calibration (if present)
            ptc = ann.get("p_team_correct")
            if isinstance(ptc, (int, float)):
                y = 1.0 if acts[(team, "decode")].correct else 0.0
                p = float(ptc)
                if 0.0 <= p <= 1.0:
                    team_brier[team].append(brier(p, y))
                    team_logloss[team].append(log_loss(p, y))

            # p_intercept: will opponent intercept this team's code?
            pi = ann.get("p_intercept")
            if isinstance(pi, (int, float)):
                p = float(pi)
                if 0.0 <= p <= 1.0:
                    # If team is RED, opponent interception is BLUE intercepting RED's code.
                    opp_team: TeamKey = "blue" if team == "red" else "red"
                    opp_intercept = acts[(opp_team, "intercept")]
                    # Round-1 (or no-evidence) intercepts are marked uninformed and excluded from ToM/calibration.
                    if not getattr(opp_intercept, "uninformed", False):
                        intercept_happened = bool(opp_intercept.correct)
                        y = 1.0 if intercept_happened else 0.0
                        opp_tom_hits[team].append(1.0 if ((p > 0.5) == intercept_happened) else 0.0)
                        leak_brier[team].append(brier(p, y))
                        leak_logloss[team].append(log_loss(p, y))

            # --- New ToM prediction tasks (24-code distributions) ---
            # (1) Predict opponent's self-decode submission (their decode consensus code)
            # (2) Predict opponent's intercept submission (their intercept consensus code)
            all_codes = [f"{a}-{b}-{c}" for a in (1, 2, 3, 4) for b in (1, 2, 3, 4) for c in (1, 2, 3, 4) if len({a, b, c}) == 3]
            opp: TeamKey = "blue" if team == "red" else "red"

            # decode dist -> compare to opponent decode consensus
            dist_dec = ann.get("opponent_decode_dist")
            if isinstance(dist_dec, dict):
                dist = _normalize_dist({k: float(v) for k, v in dist_dec.items() if isinstance(k, str) and isinstance(v, (int, float))}, all_codes)
                y_key = _code_key(acts[(opp, "decode")].consensus.guess)
                if y_key is not None:
                    tom_decode_top1[team].append(1.0 if _top1_from_dist(dist) == y_key else 0.0)
                    tom_decode_logloss[team].append(-math.log(max(1e-6, dist.get(y_key, 0.0))))

            # intercept dist -> compare to opponent intercept consensus (exclude uninformed intercept baseline)
            dist_int = ann.get("opponent_intercept_dist")
            opp_int_action = acts[(opp, "intercept")]
            if isinstance(dist_int, dict) and not getattr(opp_int_action, "uninformed", False):
                dist = _normalize_dist({k: float(v) for k, v in dist_int.items() if isinstance(k, str) and isinstance(v, (int, float))}, all_codes)
                y_key = _code_key(opp_int_action.consensus.guess)
                if y_key is not None:
                    tom_intercept_top1[team].append(1.0 if _top1_from_dist(dist) == y_key else 0.0)
                    tom_intercept_logloss[team].append(-math.log(max(1e-6, dist.get(y_key, 0.0))))

            # State-conditioned decode accuracy by clue-time state
            bucket = _bucket(team, r)
            if bucket not in by_state:
                by_state[bucket] = {"red": [], "blue": []}
            by_state[bucket][team].append(1.0 if acts[(team, "decode")].correct else 0.0)

        # --- Intercept calibration from interceptor consensus confidence ---
        for team in ("red", "blue"):
            a = acts[(team, "intercept")]
            if getattr(a, "uninformed", False):
                continue
            conf = a.consensus.confidence
            if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
                p = float(conf)
                y = 1.0 if a.correct else 0.0
                intercept_brier[team].append(brier(p, y))
                intercept_logloss[team].append(log_loss(p, y))

        # --- Overconfidence + grounding events (independent + consensus) ---
        for team in ("red", "blue"):
            for kind in ("decode", "intercept"):
                a = acts.get((team, kind))
                if a is None:
                    continue
                for ind in a.independent:
                    if getattr(ind, "overconfident", False):
                        overconf_indep[team] += 1
                    if getattr(ind, "parse_ok", True) is False:
                        parse_errors_indep[team] += 1
                    for ref in (getattr(ind, "mapping_references", []) or []):
                        if getattr(ref, "status", None) == "hypothesis":
                            hypothesis_refs_indep[team] += 1
                if getattr(a.consensus, "overconfident", False):
                    overconf_cons[team] += 1
                if getattr(a.consensus, "parse_ok", True) is False:
                    parse_errors_cons[team] += 1

                # Evaluator-side flags:
                # 1) false confirmation: agent uses 'confirmed' label without evaluator hard_confirmed support.
                # 2) within-round promotion: hypothesis -> confirmed without new evidence (i.e., still not hard_confirmed).
                cons_refs = getattr(a.consensus, "mapping_references", []) or []
                cons_status = {
                    (getattr(ref, "mapping_type", None), getattr(ref, "digit", None), getattr(ref, "value", None)): getattr(ref, "status", None)
                    for ref in cons_refs
                }

                def _is_hard_confirmed(ref) -> bool:
                    mt = getattr(ref, "mapping_type", None)
                    d = getattr(ref, "digit", None)
                    v = getattr(ref, "value", None)
                    if mt == "digit_clue" and isinstance(d, str) and isinstance(v, str):
                        return v.strip().upper() in hard_confirmed_clues[team].get(d, set())
                    # digit_theme is not hard_confirmable under core Decrypto mechanics
                    return False

                # Scan independent refs and consensus refs for false confirmations + promotions
                for ind in a.independent:
                    for ref in (getattr(ind, "mapping_references", []) or []):
                        key = (getattr(ref, "mapping_type", None), getattr(ref, "digit", None), getattr(ref, "value", None))
                        if getattr(ref, "status", None) == "hypothesis" and cons_status.get(key) == "confirmed":
                            # Promotion without new evidence if not hard-confirmed
                            if not _is_hard_confirmed(ref):
                                hypothesis_to_confirmed_promotions_within_round[team] += 1

                for ref in cons_refs:
                    if getattr(ref, "status", None) == "confirmed":
                        if not _is_hard_confirmed(ref):
                            false_confirmations_cons[team] += 1
                            # Grounding violation: claimed CONFIRMED without reveal evidence.
                            grounding_violations_cons[team] += 1
                for ind in a.independent:
                    for ref in (getattr(ind, "mapping_references", []) or []):
                        if getattr(ref, "status", None) == "confirmed":
                            if not _is_hard_confirmed(ref):
                                false_confirmations_indep[team] += 1
                                grounding_violations_indep[team] += 1

        # --- Evaluator mapping state updates (after reveal) ---
        # Update hard_confirmed digit_clue from this round's revealed codes+clues.
        for subject in ("red", "blue"):
            code = r.reveal_true_codes.get(subject)
            clueset = r.public_clues.get(subject)
            if code is None or clueset is None:
                continue
            for clue_word, d in zip(clueset.clues, code):
                ds = str(int(d))
                hard_confirmed_clues[subject][ds].add(str(clue_word).strip().upper())

        # Promote digit_theme -> soft_supported after correct decode (semantic, not hard).
        for team in ("red", "blue"):
            decoded_ok = bool(acts[(team, "decode")].correct)
            if not decoded_ok:
                continue
            priv = (r.private or {}).get(team, {})
            ann = priv.get("cluer_annotations") if isinstance(priv, dict) else None
            if not isinstance(ann, dict):
                continue
            themes = ann.get("slot_themes")
            if not isinstance(themes, dict):
                continue
            true_code = r.reveal_true_codes.get(team)
            if true_code is None:
                continue
            involved_digits = {str(int(d)) for d in true_code}
            for d in involved_digits:
                v = themes.get(d)
                if isinstance(v, str) and v.strip():
                    soft_supported_themes[team][d].add(v.strip())

        # --- Reward: hypothesis maintenance after success + hedging alignment ---
        # For each team: if they decoded correctly this round, then next round's indep digit_theme refs
        # should include multiple hypotheses (status=hypothesis) and avoid "confirmed" unless hard_confirmed.
        if idx + 1 < len(episode.rounds):
            nxt = episode.rounds[idx + 1]
            nxt_acts = _team_actions(nxt)
            for team in ("red", "blue"):
                if not bool(acts[(team, "decode")].correct):
                    continue
                # hypothesis maintenance: count distinct digit_theme hypotheses in next round (independent stage)
                hyps = 0
                for ind in nxt_acts[(team, "decode")].independent:
                    for ref in (getattr(ind, "mapping_references", []) or []):
                        if getattr(ref, "mapping_type", None) == "digit_theme" and getattr(ref, "status", None) == "hypothesis":
                            hyps += 1
                hypothesis_maintained_after_success[team].append(float(hyps))

                # hedging alignment: if a digit_theme is only soft_supported, reward using hypothesis label.
                # Compute fraction of digit_theme refs that are NOT 'confirmed' (since hard_confirmed never applies here).
                dt_refs = []
                for ref in (getattr(nxt_acts[(team, "decode")].consensus, "mapping_references", []) or []):
                    if getattr(ref, "mapping_type", None) == "digit_theme":
                        dt_refs.append(ref)
                if dt_refs:
                    ok = sum(1 for ref in dt_refs if getattr(ref, "status", None) != "confirmed")
                    hedging_alignment[team].append(ok / len(dt_refs))

        # --- Semantic slot reuse (cluer) ---
        # Compare slot_themes for consecutive rounds per team/digit.
        # Condition on opponent interception pressure and prior decode success.
        # (No embeddings; reuse is string-equality on provided theme labels.)
        if idx > 0:
            prev = episode.rounds[idx - 1]
            prev_acts = _team_actions(prev)
            for team in ("red", "blue"):
                priv_now = (r.private or {}).get(team, {})
                priv_prev = (prev.private or {}).get(team, {})
                ann_now = priv_now.get("cluer_annotations") if isinstance(priv_now, dict) else None
                ann_prev = priv_prev.get("cluer_annotations") if isinstance(priv_prev, dict) else None
                if not (isinstance(ann_now, dict) and isinstance(ann_prev, dict)):
                    continue
                st_now = ann_now.get("slot_themes")
                st_prev = ann_prev.get("slot_themes")
                if not (isinstance(st_now, dict) and isinstance(st_prev, dict)):
                    continue

                def _norm(x: Any) -> str | None:
                    if not isinstance(x, str):
                        return None
                    s = x.strip().lower()
                    return s if s else None

                # digits 1..4
                overlaps: list[float] = []
                for d in ("1", "2", "3", "4"):
                    a = _norm(st_prev.get(d))
                    b = _norm(st_now.get(d))
                    if a is None or b is None:
                        continue
                    overlaps.append(1.0 if a == b else 0.0)
                if not overlaps:
                    continue
                reuse = sum(overlaps) / len(overlaps)
                slot_reuse_overall[team].append(reuse)

                pressure = 1 if r.counters_before[team].opp_interceptions > 0 else 0
                prev_decode_ok = 1 if prev_acts[(team, "decode")].correct else 0
                bucket = f"pressure={pressure}|prev_decode_ok={prev_decode_ok}"
                if bucket not in slot_reuse_buckets:
                    slot_reuse_buckets[bucket] = {"red": [], "blue": []}
                slot_reuse_buckets[bucket][team].append(reuse)

    return {
        "tom": {
            "team_tom_accuracy": {t: _safe_mean(team_tom_hits[t]) for t in ("red", "blue")},
            "opponent_tom_accuracy": {t: _safe_mean(opp_tom_hits[t]) for t in ("red", "blue")},
            "pred_decode_top1": {t: _safe_mean(tom_decode_top1[t]) for t in ("red", "blue")},
            "pred_decode_log_loss": {t: _safe_mean(tom_decode_logloss[t]) for t in ("red", "blue")},
            "pred_intercept_top1": {t: _safe_mean(tom_intercept_top1[t]) for t in ("red", "blue")},
            "pred_intercept_log_loss": {t: _safe_mean(tom_intercept_logloss[t]) for t in ("red", "blue")},
        },
        "calibration": {
            "team_brier": {t: _safe_mean(team_brier[t]) for t in ("red", "blue")},
            "team_log_loss": {t: _safe_mean(team_logloss[t]) for t in ("red", "blue")},
            "leakage_brier": {t: _safe_mean(leak_brier[t]) for t in ("red", "blue")},
            "leakage_log_loss": {t: _safe_mean(leak_logloss[t]) for t in ("red", "blue")},
            "intercept_brier": {t: _safe_mean(intercept_brier[t]) for t in ("red", "blue")},
            "intercept_log_loss": {t: _safe_mean(intercept_logloss[t]) for t in ("red", "blue")},
            "events": {
                "overconfident_independent": dict(overconf_indep),
                "overconfident_consensus": dict(overconf_cons),
                "grounding_violations_independent": dict(grounding_violations_indep),
                "grounding_violations_consensus": dict(grounding_violations_cons),
                "parse_errors_independent": dict(parse_errors_indep),
                "parse_errors_consensus": dict(parse_errors_cons),
                "false_confirmations_independent": dict(false_confirmations_indep),
                "false_confirmations_consensus": dict(false_confirmations_cons),
                "hypothesis_refs_independent": dict(hypothesis_refs_indep),
                "hypothesis_to_confirmed_promotions_within_round": dict(hypothesis_to_confirmed_promotions_within_round),
            },
        },
        "adaptation": {
            "decode_success_by_state": {
                bucket: {t: _safe_mean(by_state[bucket][t]) for t in ("red", "blue")}
                for bucket in sorted(by_state.keys())
            },
            "semantic_slot_reuse": {
                "overall_mean": {t: _safe_mean(slot_reuse_overall[t]) for t in ("red", "blue")},
                "by_condition": {
                    bucket: {t: _safe_mean(slot_reuse_buckets[bucket][t]) for t in ("red", "blue")}
                    for bucket in sorted(slot_reuse_buckets.keys())
                },
            },
            "belief_rewards": {
                "hypotheses_after_success_mean": {t: _safe_mean(hypothesis_maintained_after_success[t]) for t in ("red", "blue")},
                "hedging_alignment_mean": {t: _safe_mean(hedging_alignment[t]) for t in ("red", "blue")},
            },
        },
    }

