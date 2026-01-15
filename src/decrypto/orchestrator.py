from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from .game import (
    check_winner,
    initial_counters,
    update_counters_after_round,
)
from .models import (
    ActionLog,
    ClueSet,
    DecryptoConfig,
    DecryptoEpisodeRecord,
    RoundCounters,
    RoundInputs,
    RoundLog,
    RoundStateTag,
    TeamKey,
)


def _state_tag(c: RoundCounters) -> RoundStateTag:
    if c.own_interceptions > c.opp_interceptions:
        st: Any = "leading"
    elif c.own_interceptions < c.opp_interceptions:
        st = "trailing"
    else:
        st = "tied"
    return RoundStateTag(
        interceptions_state=st,
        danger=(c.own_miscommunications == 1),
    )


@dataclass(frozen=True)
class TeamAgents:
    """
    Minimal agent interface for orchestrator.

    In production this will be backed by LLM agents; in tests we can pass stubs.
    """

    cluer: Any
    guesser_1: Any
    guesser_2: Any


async def _run_action(
    *,
    round_inputs: RoundInputs,
    action_kind: str,
    acting_team: TeamKey,
    opponent_team: TeamKey,
    run_action_fn: Callable[[RoundInputs, TeamKey, TeamKey, str], Awaitable[ActionLog]],
) -> ActionLog:
    return await run_action_fn(round_inputs, acting_team, opponent_team, action_kind)


async def run_round(
    *,
    base_inputs: RoundInputs,
    red_clues: ClueSet,
    blue_clues: ClueSet,
    run_action_fn: Callable[[RoundInputs, TeamKey, TeamKey, str], Awaitable[ActionLog]],
    private_cluer_fields: dict[TeamKey, dict[str, Any]] | None = None,
) -> tuple[RoundLog, dict[TeamKey, RoundCounters], TeamKey | None, str | None]:
    """
    Strict simultaneous semantics:
    - Create a frozen RoundInputs snapshot with public clues filled.
    - Run all four actions from that same snapshot (in parallel).
    - Apply exactly one reveal + counters update.
    """
    round_inputs = base_inputs.model_copy(
        update={"public_clues": {"red": red_clues, "blue": blue_clues}}
    )

    # Run 4 actions off the same snapshot (order-independent).
    red_decode_t = asyncio.create_task(
        _run_action(
            round_inputs=round_inputs,
            action_kind="decode",
            acting_team="red",
            opponent_team="blue",
            run_action_fn=run_action_fn,
        )
    )
    blue_decode_t = asyncio.create_task(
        _run_action(
            round_inputs=round_inputs,
            action_kind="decode",
            acting_team="blue",
            opponent_team="red",
            run_action_fn=run_action_fn,
        )
    )
    red_intercept_t = asyncio.create_task(
        _run_action(
            round_inputs=round_inputs,
            action_kind="intercept",
            acting_team="red",
            opponent_team="blue",
            run_action_fn=run_action_fn,
        )
    )
    blue_intercept_t = asyncio.create_task(
        _run_action(
            round_inputs=round_inputs,
            action_kind="intercept",
            acting_team="blue",
            opponent_team="red",
            run_action_fn=run_action_fn,
        )
    )

    red_decode, blue_decode, red_intercept, blue_intercept = await asyncio.gather(
        red_decode_t, blue_decode_t, red_intercept_t, blue_intercept_t
    )

    # Reveal: evaluate correctness only here (no mid-round state writes).
    true_red = base_inputs.current_codes["red"]
    true_blue = base_inputs.current_codes["blue"]

    # IMPORTANT: compute correctness here (single reveal/update), not inside run_action_fn.
    red_decode_correct = bool(red_decode.consensus.guess == true_red)
    blue_decode_correct = bool(blue_decode.consensus.guess == true_blue)
    red_intercept_correct = bool(red_intercept.consensus.guess == true_blue)
    blue_intercept_correct = bool(blue_intercept.consensus.guess == true_red)

    red_decode = red_decode.model_copy(update={"correct": red_decode_correct})
    blue_decode = blue_decode.model_copy(update={"correct": blue_decode_correct})
    red_intercept = red_intercept.model_copy(update={"correct": red_intercept_correct})
    blue_intercept = blue_intercept.model_copy(update={"correct": blue_intercept_correct})

    counters_after = update_counters_after_round(
        base_inputs.counters_before,
        red_intercept_correct=red_intercept_correct,
        blue_intercept_correct=blue_intercept_correct,
        red_decode_correct=red_decode_correct,
        blue_decode_correct=blue_decode_correct,
    )

    winner, reason = check_winner(
        counters_after,
        round_number=base_inputs.round_number,
        max_rounds=9999,  # caller decides max_rounds; checked in run_episode
    )

    round_log = RoundLog(
        round_number=base_inputs.round_number,
        counters_before=base_inputs.counters_before,
        counters_after=counters_after,
        round_state_at_clue_time={
            "red": _state_tag(base_inputs.counters_before["red"]),
            "blue": _state_tag(base_inputs.counters_before["blue"]),
        },
        public_clues={"red": red_clues, "blue": blue_clues},
        reveal_true_codes={"red": true_red, "blue": true_blue},
        actions=(red_decode, blue_decode, red_intercept, blue_intercept),
        round_state=_extract_round_state((red_decode, blue_decode, red_intercept, blue_intercept)),
        private=(private_cluer_fields or {}),
    )

    return round_log, counters_after, winner, reason


def _extract_round_state(actions: tuple[ActionLog, ActionLog, ActionLog, ActionLog]) -> dict[TeamKey, dict[str, Any]]:
    """
    Extract a structured slot-hypothesis table from agent outputs.
    This is *not* inferred from natural language; it uses the structured mapping_references / slot_hypotheses.
    """
    by_team: dict[TeamKey, dict[str, Any]] = {"red": {}, "blue": {}}
    for a in actions:
        team = a.team
        key = f"{a.kind}_consensus"
        cons = a.consensus
        # Build a slot->themes list from mapping_references digit_theme
        slot_table: dict[str, list[dict[str, Any]]] = {str(d): [] for d in range(1, 5)}
        for ref in (getattr(cons, "mapping_references", []) or []):
            if getattr(ref, "mapping_type", None) != "digit_theme":
                continue
            d = getattr(ref, "digit", None)
            v = getattr(ref, "value", None)
            if not (isinstance(d, str) and isinstance(v, str)):
                continue
            slot_table[d].append(
                {
                    "theme": v,
                    "status": getattr(ref, "status", None),
                    "support": getattr(ref, "support", None),
                    "confidence": cons.confidence,
                }
            )
        by_team[team][key] = {
            "guess": cons.guess,
            "confidence": cons.confidence,
            "slot_hypotheses": getattr(cons, "slot_hypotheses", None),
            "slot_table": slot_table,
        }
    return by_team


async def run_episode(
    *,
    config: DecryptoConfig,
    game_id: str,
    keys: dict[TeamKey, tuple[str, str, str, str]],
    code_sequences: dict[TeamKey, tuple[tuple[int, int, int], ...]],
    run_cluer_fn: Callable[[RoundInputs, TeamKey], Awaitable[tuple[ClueSet, dict[str, Any]]]],
    run_action_fn: Callable[[RoundInputs, TeamKey, TeamKey, str], Awaitable[ActionLog]],
    episode_id: str,
    timestamp: datetime | None = None,
) -> DecryptoEpisodeRecord:
    """
    Run a full Decrypto episode using strict snapshot semantics.
    """
    ts = timestamp or datetime.utcnow()
    counters = initial_counters()
    history: list[RoundLog] = []

    winner: TeamKey | None = None
    reason: str | None = None

    for r in range(1, config.max_rounds + 1):
        # Pre-generated codes per team for determinism.
        red_code = code_sequences["red"][r - 1]
        blue_code = code_sequences["blue"][r - 1]

        # Frozen base snapshot for cluer phase (no clues yet).
        base_inputs = RoundInputs(
            game_id=game_id,
            seed=config.seed,
            round_number=r,
            keys=keys,
            current_codes={"red": red_code, "blue": blue_code},
            history_rounds=tuple(history),
            counters_before=counters,
            public_clues=None,
        )

        # Cluer phase: both cluer outputs from the same base snapshot.
        (red_clues, red_private), (blue_clues, blue_private) = await asyncio.gather(
            run_cluer_fn(base_inputs, "red"),
            run_cluer_fn(base_inputs, "blue"),
        )

        # Round (strict): all four actions read only from base snapshot + public clues.
        round_log, counters, w, _unused = await run_round(
            base_inputs=base_inputs,
            red_clues=red_clues,
            blue_clues=blue_clues,
            run_action_fn=run_action_fn,
            private_cluer_fields={"red": red_private, "blue": blue_private},
        )

        history.append(round_log)

        # Terminal check after single reveal/update.
        w2, reason2 = check_winner(counters, round_number=r, max_rounds=config.max_rounds)
        winner = w2
        reason = reason2
        if winner is not None or reason in ("survived", "max_rounds"):
            break

    episode = DecryptoEpisodeRecord(
        episode_id=episode_id,
        timestamp=ts,
        config=config,
        game_id=game_id,
        seed=config.seed,
        keys=keys,
        code_sequences=code_sequences,
        rounds=tuple(history),
        winner=winner,
        result_reason=reason,  # type: ignore[arg-type]
        scores={},
    )

    # Metrics are computed after the episode is fully constructed.
    from .metrics import compute_episode_scores

    return episode.model_copy(update={"scores": compute_episode_scores(episode)})

