from __future__ import annotations

from datetime import datetime

import pytest

from src.decrypto.models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    GuesserIndependent,
    GuesserShare,
    RoundCounters,
    RoundInputs,
    RoundLog,
    RoundStateTag,
)
from src.decrypto.visibility import (
    assert_no_internal_leaks,
    assert_view_safe,
    view_for_cluer,
    view_for_guesser_decode,
    view_for_guesser_intercept,
)


def _round_inputs() -> RoundInputs:
    return RoundInputs(
        game_id="game1234",
        seed=123,
        round_number=1,
        keys={"red": ("WHALE", "CLOCK", "FOREST", "PIANO"), "blue": ("APPLE", "BREAD", "CHAIR", "DELTA")},
        current_codes={"red": (2, 4, 1), "blue": (1, 3, 4)},
        history_rounds=tuple(),
        counters_before={
            "red": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
            "blue": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
        },
        public_clues={"red": ClueSet(clues=("TICK", "KEYS", "OCEAN")), "blue": ClueSet(clues=("X", "Y", "Z"))},
    )


def _dummy_action(team: str, opponent: str, kind: str) -> ActionLog:
    ind = (
        GuesserIndependent(agent_id=f"{team}_g1", guess=(1, 2, 3), confidence=0.5, rationale="r1", parse_ok=True),
        GuesserIndependent(agent_id=f"{team}_g2", guess=(1, 2, 3), confidence=0.5, rationale="r2", parse_ok=True),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    consensus = ConsensusGuess(captain_id=f"{team}_g1", guess=(1, 2, 3), confidence=0.5, rationale="c", parse_ok=True)
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,  # type: ignore[arg-type]
        opponent_team=opponent,  # type: ignore[arg-type]
        independent=ind,
        share=share,
        consensus=consensus,
        correct=False,
    )


def test_cluer_view_includes_key_and_code_but_not_annotations_or_rng() -> None:
    ri = _round_inputs()
    payload = view_for_cluer(ri, "red")
    assert payload["role"] == "cluer"
    assert payload["team"] == "red"
    assert "key" in payload
    assert "code" in payload
    assert "annotations" not in payload
    assert "private" not in payload
    assert "rng" not in payload
    assert "confirmed_index_mappings" not in payload
    assert_view_safe(payload)


def test_decode_view_excludes_current_code_and_private_fields() -> None:
    ri = _round_inputs()
    payload = view_for_guesser_decode(ri, "red")
    assert payload["role"] == "guesser_decode"
    assert "key" in payload
    assert "code" not in payload
    assert "current_codes" not in payload
    assert "private" not in payload
    assert "annotations" not in payload
    assert "confirmed_index_mappings" not in payload
    assert_view_safe(payload)


def test_intercept_view_includes_own_key_but_not_codes_or_opponent_keys() -> None:
    ri = _round_inputs()
    payload = view_for_guesser_intercept(ri, "red")
    assert payload["role"] == "guesser_intercept"
    assert "key" in payload
    assert "keys" not in payload
    assert "code" not in payload
    assert "current_codes" not in payload
    assert "confirmed_index_mappings" not in payload
    assert_view_safe(payload)


def test_history_rounds_are_public_only() -> None:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)
    round_log = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _dummy_action("red", "blue", "decode"),
            _dummy_action("blue", "red", "decode"),
            _dummy_action("red", "blue", "intercept"),
            _dummy_action("blue", "red", "intercept"),
        ),
    )
    ri = _round_inputs().model_copy(update={"history_rounds": (round_log,)})
    payload = view_for_guesser_decode(ri, "red")
    history = payload["history_rounds"]
    assert isinstance(history, list) and len(history) == 1
    entry = history[0]
    assert "actions" not in entry
    assert "round_state" not in entry
    assert "private" not in entry
    assert "public_clues" in entry
    assert "reveal_true_codes" in entry
    assert "guesses" in entry
    assert_view_safe(payload)


def test_forbidden_field_scanner_catches_internal_state() -> None:
    with pytest.raises(AssertionError):
        assert_no_internal_leaks({"seed": 123})
