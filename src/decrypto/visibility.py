from __future__ import annotations

from typing import Any

from .models import RoundInputs, RoundLog, TeamKey


FORBIDDEN_INTERNAL_KEYS = {
    # internal state / debug that must never be in any agent view
    "seed",
    "rng",
    "random",
    "debug",
    "annotations",
    "cluer_annotations",
    "private",
    "actions",
    "round_state",
    # raw code container should never be present; role views should be explicit
    "current_codes",
    "codes",
}

ROLE_FORBIDDEN_KEYS = {
    "cluer": set(),  # cluer is allowed to see own key + own code
    "guesser_decode": {"code", "current_codes"},
    "guesser_intercept": {"keys", "code", "current_codes"},
}


def _public_round_summary(round_log: RoundLog) -> dict[str, Any]:
    """
    Public round view only: clues, true codes, and final guesses.
    Deliberations and private annotations are excluded.
    """
    actions = {(a.team, a.kind): a for a in round_log.actions}
    guesses: dict[str, dict[str, Any]] = {}
    for team, kind in (("red", "decode"), ("blue", "decode"), ("red", "intercept"), ("blue", "intercept")):
        action = actions.get((team, kind))
        if action is None:
            continue
        guess = action.consensus.guess
        guesses[f"{team}_{kind}"] = {
            "team": team,
            "kind": kind,
            "guess": list(guess) if guess is not None else None,
            "correct": bool(action.correct),
        }

    return {
        "round_number": round_log.round_number,
        "public_clues": {k: {"clues": list(v.clues)} for k, v in round_log.public_clues.items()},
        "reveal_true_codes": {k: list(v) for k, v in round_log.reveal_true_codes.items()},
        "guesses": guesses,
    }


def view_for_cluer(round_inputs: RoundInputs, team: TeamKey) -> dict[str, Any]:
    """
    Cluer sees:
    - own key
    - own current code digits
    - full revealed history (both teams)
    - public counters
    - no private annotations (ever)
    """
    opp: TeamKey = "blue" if team == "red" else "red"
    return {
        "role": "cluer",
        "team": team,
        "opponent_team": opp,
        "key": list(round_inputs.keys[team]),
        "code": list(round_inputs.current_codes[team]),
        # History must be public-only: clues, true codes, and final guesses only.
        "history_rounds": [_public_round_summary(r) for r in round_inputs.history_rounds],
        "game_state": round_inputs.counters_before[team].model_dump(mode="json"),
        "opponent_game_state": round_inputs.counters_before[opp].model_dump(mode="json"),
        "public_clues": (
            None
            if round_inputs.public_clues is None
            else {k: list(v.clues) for k, v in round_inputs.public_clues.items()}
        ),
    }


def view_for_guesser_decode(round_inputs: RoundInputs, team: TeamKey) -> dict[str, Any]:
    """
    Own decode guessers see:
    - own key
    - public clues (both teams, once available)
    - full revealed history
    - public counters
    Must NOT see:
    - own current code digits
    - any private annotations
    """
    opp: TeamKey = "blue" if team == "red" else "red"
    return {
        "role": "guesser_decode",
        "team": team,
        "opponent_team": opp,
        "key": list(round_inputs.keys[team]),
        "history_rounds": [_public_round_summary(r) for r in round_inputs.history_rounds],
        "game_state": round_inputs.counters_before[team].model_dump(mode="json"),
        "opponent_game_state": round_inputs.counters_before[opp].model_dump(mode="json"),
        "public_clues": (
            None
            if round_inputs.public_clues is None
            else {k: list(v.clues) for k, v in round_inputs.public_clues.items()}
        ),
    }


def view_for_guesser_intercept(round_inputs: RoundInputs, team: TeamKey) -> dict[str, Any]:
    """
    Opponent intercept guessers see:
    - own key (opponent key remains hidden)
    - public clues (both teams, once available)
    - full revealed history
    - public counters
    Must NOT see:
    - opponent key words
    - any current code digits
    - any private annotations
    """
    opp: TeamKey = "blue" if team == "red" else "red"
    return {
        "role": "guesser_intercept",
        "team": team,
        "opponent_team": opp,
        "key": list(round_inputs.keys[team]),
        "history_rounds": [_public_round_summary(r) for r in round_inputs.history_rounds],
        "game_state": round_inputs.counters_before[team].model_dump(mode="json"),
        "opponent_game_state": round_inputs.counters_before[opp].model_dump(mode="json"),
        "public_clues": (
            None
            if round_inputs.public_clues is None
            else {k: list(v.clues) for k, v in round_inputs.public_clues.items()}
        ),
    }


def assert_no_internal_leaks(payload: Any) -> None:
    """
    Recursively assert that internal/debug forbidden fields do not appear in a payload.
    """
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(k, str) and k in FORBIDDEN_INTERNAL_KEYS:
                raise AssertionError(f"Forbidden internal field present in payload: {k}")
            assert_no_internal_leaks(v)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_internal_leaks(item)


def assert_view_safe(payload: Any) -> None:
    """
    Validate a view payload for leakage based on its declared role.
    """
    if not isinstance(payload, dict):
        raise AssertionError("View payload must be a dict")
    role = payload.get("role")
    if role not in ROLE_FORBIDDEN_KEYS:
        raise AssertionError(f"Unknown role in view payload: {role!r}")

    assert_no_internal_leaks(payload)
    forbidden = ROLE_FORBIDDEN_KEYS[role]  # type: ignore[index]
    for k in forbidden:
        if k in payload:
            raise AssertionError(f"Forbidden field for role={role}: {k}")
