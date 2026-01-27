from __future__ import annotations

import pytest

from src.agents.llm import MockProvider
from src.decrypto.agents.llm_agents import DecryptoGuesserLLM
from src.decrypto.models import ClueSet, RoundCounters, RoundInputs


def _round_inputs() -> RoundInputs:
    return RoundInputs(
        game_id="g",
        seed=1,
        round_number=1,
        keys={"red": ("WHALE", "CLOCK", "FOREST", "PIANO"), "blue": ("A", "B", "C", "D")},
        current_codes={"red": (2, 4, 1), "blue": (1, 3, 4)},
        history_rounds=tuple(),
        counters_before={
            "red": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
            "blue": RoundCounters(own_interceptions=0, own_miscommunications=0, opp_interceptions=0, opp_miscommunications=0),
        },
        public_clues={"red": ClueSet(clues=("TICK", "KEYS", "OCEAN")), "blue": ClueSet(clues=("X", "Y", "Z"))},
    )


@pytest.mark.asyncio
async def test_independent_guess_single_repair_retry_then_success() -> None:
    provider = MockProvider(
        responses=[
            "NOT JSON",
            "GUESS=(1,2,3) CONF=0.8",
        ]
    )
    g = DecryptoGuesserLLM(provider=provider, agent_id="red_g1", team="red")
    out, scratchpad = await g.independent_guess(_round_inputs(), "decode")
    assert out.parse_ok is True
    assert out.guess == (1, 2, 3)
    assert out.confidence == 0.8
    assert out.parse_retry_used is True
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_independent_guess_failure_after_repair() -> None:
    provider = MockProvider(responses=["NOPE", "STILL NOPE"])
    g = DecryptoGuesserLLM(provider=provider, agent_id="red_g1", team="red")
    out, scratchpad = await g.independent_guess(_round_inputs(), "decode")
    assert out.parse_ok is False
    # No abstentions: should fall back to deterministic guess
    assert out.guess == (1, 2, 3)
    assert out.confidence == 0.0
    assert provider.call_count == 2
