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
    RoundLog,
    RoundStateTag,
    RoundInputs,
)


def _history_round_with_private_key_leak() -> RoundLog:
    """
    History round includes private cluer annotations that reference a key word.
    Visibility builders must exclude this from all role payloads.
    """
    dummy_action = ActionLog(
        kind="decode",
        team="red",
        opponent_team="blue",
        independent=(
            GuesserIndependent(agent_id="a", guess=(1, 2, 3), confidence=0.5, rationale="x", parse_ok=True),
            GuesserIndependent(agent_id="b", guess=(1, 2, 3), confidence=0.5, rationale="y", parse_ok=True),
        ),
        share=(GuesserShare(agent_id="a", message="m"), GuesserShare(agent_id="b", message="n")),
        consensus=ConsensusGuess(captain_id="a", guess=(1, 2, 3), confidence=0.5, rationale="z", parse_ok=True),
        correct=False,
    )
    c0 = RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0)
    tag = RoundStateTag(interceptions_state="tied", danger=False)
    return RoundLog(
        round_number=1,
        counters_before={"red": c0, "blue": c0},
        counters_after={"red": c0, "blue": c0},
        round_state_at_clue_time={"red": tag, "blue": tag},
        public_clues={"red": ClueSet(clues=("A", "B", "C")), "blue": ClueSet(clues=("D", "E", "F"))},
        reveal_true_codes={"red": (1, 2, 3), "blue": (2, 3, 4)},
        actions=(dummy_action, dummy_action, dummy_action, dummy_action),
        private={"red": {"cluer_annotations": {"intended_mapping": {"1": "WHALE"}}}},
    )


def _round_inputs() -> RoundInputs:
    return RoundInputs(
        game_id="g",
        seed=123,
        round_number=2,
        keys={"red": ("WHALE", "CLOCK", "FOREST", "PIANO"), "blue": ("APPLE", "BREAD", "CHAIR", "DELTA")},
        current_codes={"red": (2, 4, 1), "blue": (1, 3, 4)},
        history_rounds=(_history_round_with_private_key_leak(),),
        counters_before={
            "red": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
            "blue": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
        },
        public_clues={"red": ClueSet(clues=("TICK", "KEYS", "OCEAN")), "blue": ClueSet(clues=("X", "Y", "Z"))},
    )


@pytest.mark.asyncio
async def test_intercept_prompt_excludes_opponent_key_and_private_history() -> None:
    provider = MockProvider(
        responses=['{"guess":[1,2,3],"confidence":0.5,"rationale":"x"}']
    )
    g = DecryptoGuesserLLM(provider=provider, agent_id="red_g1", team="red")
    await g.independent_guess(_round_inputs(), "intercept")

    # Scan prompt content given to the LLM.
    full_prompt = "\n".join(m["content"] for m in provider.last_messages)

    # Must not include opponent key words, any private fields, or current code digits.
    for forbidden in ["APPLE", "BREAD", "CHAIR", "DELTA"]:
        assert forbidden not in full_prompt
    for forbidden in ["cluer_annotations", "private", "actions", "round_state"]:
        assert forbidden not in full_prompt
    assert "2, 4, 1" not in full_prompt
    assert "1, 3, 4" not in full_prompt
