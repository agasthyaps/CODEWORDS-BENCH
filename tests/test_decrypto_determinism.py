from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.decrypto.game import create_game
from src.decrypto.models import (
    ActionLog,
    ClueSet,
    ConsensusGuess,
    DecryptoConfig,
    GuesserIndependent,
    GuesserShare,
    RoundInputs,
)
from src.decrypto.orchestrator import run_episode


async def _stub_run_cluer_fn(round_inputs: RoundInputs, team: str):
    # Deterministic clues + private annotations.
    clues = ClueSet(clues=("A", "B", "C")) if team == "red" else ClueSet(clues=("D", "E", "F"))
    ann = {
        "intended_mapping": {"1": "X", "2": "Y", "3": "Z"},
        "predicted_team_guess": [1, 2, 3],
        "p_team_correct": 0.6,
        "p_intercept": 0.2,
    }
    return clues, {"cluer_annotations": ann}


def _action_log(team: str, opponent: str, kind: str, guess: tuple[int, int, int]) -> ActionLog:
    ind = (
        GuesserIndependent(agent_id=f"{team}_g1", guess=guess, confidence=0.7, rationale="r1", parse_ok=True),
        GuesserIndependent(agent_id=f"{team}_g2", guess=guess, confidence=0.6, rationale="r2", parse_ok=True),
    )
    share = (
        GuesserShare(agent_id=f"{team}_g1", message="m1"),
        GuesserShare(agent_id=f"{team}_g2", message="m2"),
    )
    consensus = ConsensusGuess(captain_id=f"{team}_g1", guess=guess, confidence=0.65, rationale="c", parse_ok=True)
    return ActionLog(
        kind=kind,  # type: ignore[arg-type]
        team=team,  # type: ignore[arg-type]
        opponent_team=opponent,  # type: ignore[arg-type]
        independent=ind,
        share=share,
        consensus=consensus,
        correct=False,
        confirmed_mapping_count=0,
        uninformed=False,
    )


async def _stub_run_action_fn(round_inputs: RoundInputs, team: str, opponent: str, kind: str) -> ActionLog:
    # Deterministic: always guess (1,2,3) regardless of view.
    return _action_log(team, opponent, kind, (1, 2, 3))


@pytest.mark.asyncio
async def test_episode_determinism_same_inputs_same_outputs() -> None:
    cfg = DecryptoConfig(max_rounds=2, seed=42)
    _game_id, _seed, keys, code_sequences = create_game(cfg)

    fixed_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)

    ep1 = await run_episode(
        config=cfg,
        game_id="fixedgame",
        keys=keys,
        code_sequences=code_sequences,
        run_cluer_fn=_stub_run_cluer_fn,
        run_action_fn=_stub_run_action_fn,
        episode_id="epfixed",
        timestamp=fixed_ts,
    )
    ep2 = await run_episode(
        config=cfg,
        game_id="fixedgame",
        keys=keys,
        code_sequences=code_sequences,
        run_cluer_fn=_stub_run_cluer_fn,
        run_action_fn=_stub_run_action_fn,
        episode_id="epfixed",
        timestamp=fixed_ts,
    )

    assert ep1.model_dump(mode="json") == ep2.model_dump(mode="json")

