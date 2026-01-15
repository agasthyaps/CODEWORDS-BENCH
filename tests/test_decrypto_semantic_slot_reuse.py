from __future__ import annotations

from datetime import datetime, timezone

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


def _simple_action(team: str, opp: str, kind: str) -> ActionLog:
    ind = (
        GuesserIndependent(agent_id=f"{team}_g1", guess=(1, 2, 3), confidence=0.5, rationale="speculative", parse_ok=True),
        GuesserIndependent(agent_id=f"{team}_g2", guess=(1, 2, 3), confidence=0.5, rationale="speculative", parse_ok=True),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    cons = ConsensusGuess(captain_id=f"{team}_g1", guess=(1, 2, 3), confidence=0.5, rationale="speculative", parse_ok=True)
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,  # type: ignore[arg-type]
        opponent_team=opp,  # type: ignore[arg-type]
        independent=ind,
        share=share,
        consensus=cons,
        correct=False,
        confirmed_mapping_count=1,
        uninformed=False,
    )


def test_semantic_slot_reuse_computed_from_cluer_slot_themes() -> None:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    c_pressure = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=1, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)

    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _simple_action("red", "blue", "decode"),
            _simple_action("blue", "red", "decode"),
            _simple_action("red", "blue", "intercept"),
            _simple_action("blue", "red", "intercept"),
        ),
        private={
            "red": {"cluer_annotations": {"slot_themes": {"1": "France", "2": "Dwarf", "3": "Chocolate", "4": "Ocean"}}},
            "blue": {"cluer_annotations": {"slot_themes": {"1": "Music", "2": "Time", "3": "Forest", "4": "Whale"}}},
        },
    )

    # Round 2: RED repeats two themes, changes two; BLUE repeats all.
    r2 = RoundLog(
        round_number=2,
        counters_before={"red": c_pressure, "blue": c0},  # pressure for red only
        counters_after={"red": c_pressure, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("G", "H", "I")), "blue": ClueSet(clues=("J", "K", "L"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _simple_action("red", "blue", "decode"),
            _simple_action("blue", "red", "decode"),
            _simple_action("red", "blue", "intercept"),
            _simple_action("blue", "red", "intercept"),
        ),
        private={
            "red": {"cluer_annotations": {"slot_themes": {"1": "France", "2": "Dwarf", "3": "Candy", "4": "Sea"}}},
            "blue": {"cluer_annotations": {"slot_themes": {"1": "Music", "2": "Time", "3": "Forest", "4": "Whale"}}},
        },
    )

    episode = DecryptoEpisodeRecord(
        episode_id="reuse",
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        config=DecryptoConfig(seed=0, max_rounds=2),
        game_id="g",
        seed=0,
        keys={"red": ("W", "X", "Y", "Z"), "blue": ("A", "B", "C", "D")},
        code_sequences={"red": ((1, 2, 3), (1, 2, 3)), "blue": ((2, 3, 4), (2, 3, 4))},
        rounds=(r1, r2),
        winner=None,
        result_reason="survived",
        scores={},
    )

    scores = compute_episode_scores(episode)
    reuse = scores["adaptation"]["semantic_slot_reuse"]["overall_mean"]
    # RED reuse: 2/4 digits repeated => 0.5; BLUE reuse 4/4 => 1.0
    assert reuse["red"] == 0.5
    assert reuse["blue"] == 1.0

