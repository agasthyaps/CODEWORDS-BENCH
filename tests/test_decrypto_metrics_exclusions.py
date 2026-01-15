from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.decrypto.metrics import compute_episode_scores
from src.decrypto.models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    DecryptoConfig,
    DecryptoEpisodeRecord,
    GuesserIndependent,
    GuesserShare,
    RoundCounters,
    RoundLog,
    RoundStateTag,
)


def _action(team: str, opp: str, kind: str, *, guess, conf, correct, uninformed=False, overconf=False) -> ActionLog:
    ind = (
        GuesserIndependent(agent_id=f"{team}_g1", guess=guess, confidence=conf, rationale="speculative", parse_ok=True, overconfident=overconf),
        GuesserIndependent(agent_id=f"{team}_g2", guess=guess, confidence=conf, rationale="speculative", parse_ok=True, overconfident=overconf),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    cons = ConsensusGuess(
        captain_id=f"{team}_g1",
        guess=guess,
        confidence=conf,
        rationale="speculative",
        parse_ok=True,
        overconfident=overconf,
    )
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,  # type: ignore[arg-type]
        opponent_team=opp,  # type: ignore[arg-type]
        independent=ind,
        share=share,
        consensus=cons,
        correct=correct,
        confirmed_mapping_count=0,
        uninformed=uninformed,
    )


def test_uninformed_intercepts_excluded_from_opponent_tom_and_intercept_calibration() -> None:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)

    # Round 1: opponent intercepts are uninformed
    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _action("red", "blue", "decode", guess=(1, 2, 3), conf=0.6, correct=True),
            _action("blue", "red", "decode", guess=(2, 3, 4), conf=0.6, correct=True),
            _action("red", "blue", "intercept", guess=(2, 3, 4), conf=0.9, correct=True, uninformed=True),
            _action("blue", "red", "intercept", guess=(1, 2, 3), conf=0.9, correct=True, uninformed=True),
        ),
        private={
            "red": {"cluer_annotations": {"p_intercept": 0.9, "p_team_correct": 0.5, "predicted_team_guess": [1, 2, 3]}},
            "blue": {"cluer_annotations": {"p_intercept": 0.9, "p_team_correct": 0.5, "predicted_team_guess": [2, 3, 4]}},
        },
    )

    episode = DecryptoEpisodeRecord(
        episode_id="e",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        config=DecryptoConfig(seed=0, max_rounds=1),
        game_id="g",
        seed=0,
        keys={"red": ("W", "X", "Y", "Z"), "blue": ("A", "B", "C", "D")},
        code_sequences={"red": ((1, 2, 3),), "blue": ((2, 3, 4),)},
        rounds=(r1,),
        winner=None,
        result_reason="survived",
        scores={},
    )

    scores = compute_episode_scores(episode)

    # Opponent-ToM should be None (no scored points) because opponent intercepts are uninformed.
    assert scores["tom"]["opponent_tom_accuracy"]["red"] is None
    assert scores["tom"]["opponent_tom_accuracy"]["blue"] is None

    # Intercept calibration should also be None because intercept actions were uninformed.
    assert scores["calibration"]["intercept_brier"]["red"] is None
    assert scores["calibration"]["intercept_brier"]["blue"] is None


def test_overconfident_event_counted_when_no_confirmed_mappings() -> None:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)

    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _action("red", "blue", "decode", guess=(1, 2, 3), conf=0.95, correct=False, overconf=True),
            _action("blue", "red", "decode", guess=(2, 3, 4), conf=0.95, correct=False, overconf=True),
            _action("red", "blue", "intercept", guess=(2, 3, 4), conf=0.95, correct=False, uninformed=True, overconf=True),
            _action("blue", "red", "intercept", guess=(1, 2, 3), conf=0.95, correct=False, uninformed=True, overconf=True),
        ),
        private={"red": {"cluer_annotations": {}}, "blue": {"cluer_annotations": {}}},
    )

    episode = DecryptoEpisodeRecord(
        episode_id="e2",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        config=DecryptoConfig(seed=0, max_rounds=1),
        game_id="g",
        seed=0,
        keys={"red": ("W", "X", "Y", "Z"), "blue": ("A", "B", "C", "D")},
        code_sequences={"red": ((1, 2, 3),), "blue": ((2, 3, 4),)},
        rounds=(r1,),
        winner=None,
        result_reason="survived",
        scores={},
    )

    scores = compute_episode_scores(episode)
    # Each team has 2 independent agents * 2 actions (decode+intercept) = 4 overconfident indep events.
    assert scores["calibration"]["events"]["overconfident_independent"]["red"] == 4
    assert scores["calibration"]["events"]["overconfident_independent"]["blue"] == 4
    # Each team has 2 consensus events (decode+intercept).
    assert scores["calibration"]["events"]["overconfident_consensus"]["red"] == 2
    assert scores["calibration"]["events"]["overconfident_consensus"]["blue"] == 2


def test_false_confirmations_counted_when_mapping_labels_invalid() -> None:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)

    # Under the new schema, false confirmations are evaluated evaluator-side based on mapping_references.
    # Create a decode action that claims a digit_theme is CONFIRMED, which cannot be hard-confirmed in core Decrypto.
    from src.decrypto.models import MappingReference
    bad_ind = (
        GuesserIndependent(
            agent_id="red_g1",
            guess=(1, 2, 3),
            confidence=0.6,
            rationale="speculative",
            parse_ok=True,
            mapping_references=[MappingReference(mapping_type="digit_theme", digit="1", value="FRANCE", status="confirmed")],
        ),
        GuesserIndependent(
            agent_id="red_g2",
            guess=(1, 2, 3),
            confidence=0.6,
            rationale="speculative",
            parse_ok=True,
            mapping_references=[MappingReference(mapping_type="digit_theme", digit="1", value="FRANCE", status="confirmed")],
        ),
    )
    share = (GuesserShare(agent_id="red_g1", message="m1"), GuesserShare(agent_id="red_g2", message="m2"))
    bad_cons = ConsensusGuess(
        captain_id="red_g1",
        guess=(1, 2, 3),
        confidence=0.6,
        rationale="speculative",
        parse_ok=True,
        mapping_references=[MappingReference(mapping_type="digit_theme", digit="1", value="FRANCE", status="confirmed")],
    )
    bad_action = ActionLog(
        kind="decode",
        team="red",
        opponent_team="blue",
        independent=bad_ind,
        share=share,
        consensus=bad_cons,
        correct=False,
        confirmed_mapping_count=0,
        uninformed=False,
    )

    good_action = _action("blue", "red", "decode", guess=(2, 3, 4), conf=0.6, correct=True)
    red_int = _action("red", "blue", "intercept", guess=(2, 3, 4), conf=0.6, correct=False, uninformed=True)
    blue_int = _action("blue", "red", "intercept", guess=(1, 2, 3), conf=0.6, correct=False, uninformed=True)

    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(bad_action, good_action, red_int, blue_int),
        private={"red": {"cluer_annotations": {}}, "blue": {"cluer_annotations": {}}},
    )

    episode = DecryptoEpisodeRecord(
        episode_id="e3",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        config=DecryptoConfig(seed=0, max_rounds=1),
        game_id="g",
        seed=0,
        keys={"red": ("W", "X", "Y", "Z"), "blue": ("A", "B", "C", "D")},
        code_sequences={"red": ((1, 2, 3),), "blue": ((2, 3, 4),)},
        rounds=(r1,),
        winner=None,
        result_reason="survived",
        scores={},
    )

    scores = compute_episode_scores(episode)
    assert scores["calibration"]["events"]["false_confirmations_independent"]["red"] == 2
    assert scores["calibration"]["events"]["false_confirmations_consensus"]["red"] == 1


def test_predicted_distributions_score_top1_and_logloss() -> None:
    # Build a minimal 1-round episode where RED cluer predicts BLUE decode distribution.
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)

    # BLUE actually decodes to (2,3,4)
    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _action("red", "blue", "decode", guess=(1, 2, 3), conf=0.6, correct=True),
            _action("blue", "red", "decode", guess=(2, 3, 4), conf=0.6, correct=True),
            _action("red", "blue", "intercept", guess=(2, 3, 4), conf=0.6, correct=False, uninformed=True),
            _action("blue", "red", "intercept", guess=(1, 2, 3), conf=0.6, correct=False, uninformed=True),
        ),
        private={
            "red": {"cluer_annotations": {"opponent_decode_dist": {"2-3-4": 0.9, "1-2-3": 0.1}}},
            "blue": {"cluer_annotations": {}},
        },
    )

    episode = DecryptoEpisodeRecord(
        episode_id="pred",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        config=DecryptoConfig(seed=0, max_rounds=1),
        game_id="g",
        seed=0,
        keys={"red": ("W", "X", "Y", "Z"), "blue": ("A", "B", "C", "D")},
        code_sequences={"red": ((1, 2, 3),), "blue": ((2, 3, 4),)},
        rounds=(r1,),
        winner=None,
        result_reason="survived",
        scores={},
    )

    scores = compute_episode_scores(episode)
    assert scores["tom"]["pred_decode_top1"]["red"] == 1.0
    assert scores["tom"]["pred_decode_log_loss"]["red"] is not None

