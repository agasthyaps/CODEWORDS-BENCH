from __future__ import annotations

import pytest

from src.decrypto.game import initial_counters
from src.decrypto.metrics import code_distance, compute_adaptation_rate
from src.decrypto.models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    DecryptoConfig,
    DecryptoEpisodeRecord,
    GuesserIndependent,
    GuesserShare,
    RoundLog,
    RoundStateTag,
    TeamKey,
)


def _consensus(team: TeamKey, guess: tuple[int, int, int] | None) -> ConsensusGuess:
    return ConsensusGuess(
        captain_id=f"{team}_g1",
        guess=guess,
        confidence=0.7 if guess is not None else None,
        rationale="test",
        parse_ok=guess is not None,
    )


def _action(
    *,
    kind: str,
    team: TeamKey,
    opponent: TeamKey,
    guess: tuple[int, int, int] | None,
    correct: bool = False,
    uninformed: bool = False,
) -> ActionLog:
    independent = (
        GuesserIndependent(
            agent_id=f"{team}_g1",
            guess=guess,
            confidence=0.7 if guess is not None else None,
            rationale="r1",
            parse_ok=guess is not None,
        ),
        GuesserIndependent(
            agent_id=f"{team}_g2",
            guess=guess,
            confidence=0.6 if guess is not None else None,
            rationale="r2",
            parse_ok=guess is not None,
        ),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,
        opponent_team=opponent,
        independent=independent,
        share=share,
        consensus=_consensus(team, guess),
        correct=correct,
        uninformed=uninformed,
    )


def _round(
    *,
    round_number: int,
    true_red: tuple[int, int, int],
    true_blue: tuple[int, int, int],
    red_intercept_guess: tuple[int, int, int] | None,
    blue_intercept_guess: tuple[int, int, int] | None = (1, 2, 3),
    red_intercept_correct: bool = False,
    blue_intercept_correct: bool = False,
    red_uninformed: bool = False,
    blue_uninformed: bool = False,
) -> RoundLog:
    counters = initial_counters()
    state_tag = RoundStateTag(interceptions_state="tied", danger=False)
    return RoundLog(
        round_number=round_number,
        counters_before=counters,
        counters_after=counters,
        round_state_at_clue_time={"red": state_tag, "blue": state_tag},
        public_clues={
            "red": ClueSet(clues=("A", "B", "C")),
            "blue": ClueSet(clues=("D", "E", "F")),
        },
        reveal_true_codes={"red": true_red, "blue": true_blue},
        actions=(
            _action(kind="decode", team="red", opponent="blue", guess=true_red, correct=True),
            _action(kind="decode", team="blue", opponent="red", guess=true_blue, correct=True),
            _action(
                kind="intercept",
                team="red",
                opponent="blue",
                guess=red_intercept_guess,
                correct=red_intercept_correct,
                uninformed=red_uninformed,
            ),
            _action(
                kind="intercept",
                team="blue",
                opponent="red",
                guess=blue_intercept_guess,
                correct=blue_intercept_correct,
                uninformed=blue_uninformed,
            ),
        ),
    )


def _episode(rounds: tuple[RoundLog, ...]) -> DecryptoEpisodeRecord:
    return DecryptoEpisodeRecord(
        episode_id="adaptation-test",
        config=DecryptoConfig(max_rounds=len(rounds), seed=7),
        game_id="game-1",
        seed=7,
        keys={
            "red": ("R1", "R2", "R3", "R4"),
            "blue": ("B1", "B2", "B3", "B4"),
        },
        code_sequences={
            "red": tuple(r.reveal_true_codes["red"] for r in rounds),
            "blue": tuple(r.reveal_true_codes["blue"] for r in rounds),
        },
        rounds=rounds,
        winner=None,
        result_reason=None,
    )


def test_compute_adaptation_rate_uses_revealed_true_codes() -> None:
    episode = _episode(
        (
            _round(
                round_number=1,
                true_red=(2, 3, 4),
                true_blue=(1, 2, 3),
                red_intercept_guess=(4, 3, 2),
                red_uninformed=True,
            ),
            _round(
                round_number=2,
                true_red=(2, 3, 4),
                true_blue=(1, 2, 3),
                red_intercept_guess=(1, 4, 3),
                red_intercept_correct=False,
            ),
            _round(
                round_number=3,
                true_red=(2, 3, 4),
                true_blue=(1, 2, 3),
                red_intercept_guess=(1, 2, 3),
                red_intercept_correct=True,
            ),
        )
    )

    metrics = compute_adaptation_rate(episode)

    assert 1 not in metrics["per_round_distance"]["red"]
    assert metrics["per_round_distance"]["red"][2] == pytest.approx(code_distance((1, 4, 3), (1, 2, 3)))
    assert metrics["per_round_distance"]["red"][3] == pytest.approx(0.0)
    assert metrics["improvement_slope"]["red"] is not None
    assert metrics["improvement_slope"]["red"] < 0
    assert metrics["rounds_to_first_intercept"]["red"] == 3


def test_adaptation_does_not_depend_on_nonexistent_clue_actions() -> None:
    episode = _episode(
        (
            _round(
                round_number=1,
                true_red=(4, 3, 2),
                true_blue=(1, 2, 3),
                red_intercept_guess=(4, 3, 2),
                red_uninformed=True,
            ),
            _round(
                round_number=2,
                true_red=(4, 3, 2),
                true_blue=(1, 2, 3),
                red_intercept_guess=(1, 2, 4),
                red_intercept_correct=False,
            ),
        )
    )

    metrics = compute_adaptation_rate(episode)

    # If compute_adaptation_rate regresses to looking for `(opponent, "clue")`,
    # this distance will disappear because round actions only include decode/intercept.
    assert metrics["per_round_distance"]["red"][2] == pytest.approx(code_distance((1, 2, 4), (1, 2, 3)))
