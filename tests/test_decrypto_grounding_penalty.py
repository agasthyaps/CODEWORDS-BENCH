from __future__ import annotations

import pytest

from src.agents.llm import MockProvider
from src.decrypto.agents.llm_agents import DecryptoGuesserLLM
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


def _action(team: str, opp: str, kind: str, correct: bool) -> ActionLog:
    ind = (
        GuesserIndependent(agent_id=f"{team}_g1", guess=(1, 2, 3), confidence=0.5, rationale="speculative", parse_ok=True),
        GuesserIndependent(agent_id=f"{team}_g2", guess=(1, 2, 3), confidence=0.5, rationale="speculative", parse_ok=True),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    cons = ConsensusGuess(
        captain_id=f"{team}_g1",
        guess=(1, 2, 3),
        confidence=0.5,
        rationale="speculative",
        parse_ok=True,
    )
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,  # type: ignore[arg-type]
        opponent_team=opp,  # type: ignore[arg-type]
        independent=ind,
        share=share,
        consensus=cons,
        correct=correct,
    )


def _round_inputs_no_event_for_red_decode() -> RoundInputs:
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)
    r1 = RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("ALPHA", "BETA", "GAMMA")), "blue": ClueSet(clues=("DELTA", "EPSILON", "ZETA"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(
            _action("red", "blue", "decode", correct=False),
            _action("blue", "red", "decode", correct=True),
            _action("red", "blue", "intercept", correct=False),
            _action("blue", "red", "intercept", correct=False),
        ),
    )
    return RoundInputs(
        game_id="g",
        seed=1,
        round_number=2,
        keys={"red": ("WHALE", "CLOCK", "FOREST", "PIANO"), "blue": ("APPLE", "BREAD", "CHAIR", "DELTA")},
        current_codes={"red": (2, 4, 1), "blue": (1, 3, 4)},
        history_rounds=(r1,),
        counters_before={"red": c0, "blue": c0},
        public_clues={"red": ClueSet(clues=("TICK", "KEYS", "OCEAN")), "blue": ClueSet(clues=("X", "Y", "Z"))},
    )


@pytest.mark.asyncio
async def test_grounding_violation_penalizes_confidence() -> None:
    provider = MockProvider(
        responses=[
            (
                "{"
                "\"guess\":[1,2,3],"
                "\"confidence\":0.8,"
                "\"rationale\":\"We know this is confirmed.\","
                "\"mapping_references\":["
                "{\"mapping_type\":\"digit_clue\",\"digit\":\"1\",\"value\":\"ALPHA\",\"status\":\"confirmed\"}"
                "]"
                "}"
            )
        ]
    )
    g = DecryptoGuesserLLM(provider=provider, agent_id="red_g1", team="red")
    out = await g.independent_guess(_round_inputs_no_event_for_red_decode(), "decode")
    assert out.grounding_ok is False
    assert out.confidence is not None
    assert out.confidence < 0.5
