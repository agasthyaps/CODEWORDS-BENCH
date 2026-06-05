"""Orchestrator for running Hanabi games."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable

from src.core.state import AgentStateManager
from src.core.benchmarking import (
    BenchmarkCondition,
    BenchmarkPenalty,
    HarnessEvent,
    RunManifest,
    build_run_manifest,
    default_benchmark_condition,
    resolve_condition,
)

from .agents.llm_agent import HanabiPlayerLLM
from .game import apply_action, check_terminal, create_game
from .metrics import compute_episode_metrics
from .models import (
    Action,
    HanabiConfig,
    HanabiEpisodeRecord,
    HanabiState,
    TurnLog,
)
from .visibility import view_for_player


async def run_turn(
    state: HanabiState,
    player: HanabiPlayerLLM,
    agent_states: AgentStateManager | None,
    condition: BenchmarkCondition | None = None,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    episode_id: str | None = None,
    harness_events: list[HarnessEvent] | None = None,
    benchmark_penalties: list[BenchmarkPenalty] | None = None,
) -> tuple[HanabiState, TurnLog, str | None]:
    """
    Execute a single turn.
    
    Args:
        state: Current game state
        player: The player whose turn it is
        agent_states: Manager for agent scratchpads
        emit_fn: Optional callback for emitting events
        
    Returns:
        (new_state, turn_log, scratchpad_addition)
    """
    player_id = player.player_id
    condition = condition or default_benchmark_condition()
    scratchpad = agent_states.get_scratchpad(player_id) if agent_states else ""
    
    # Get player's action
    action, rationale, scratchpad_add = await player.decide_action(state, scratchpad)
    
    # Update scratchpad
    if agent_states is not None and scratchpad_add:
        agent_state = agent_states.get_or_create(player_id)
        agent_state.append_to_scratchpad(state.turn_number, scratchpad_add)
    
    # Apply action
    invalid_policy = (
        "human_table"
        if condition.invalid_action_policy == "human_table"
        else "no_op"
    )
    new_state, result, turn_log = apply_action(
        state, player_id, action, rationale, invalid_policy=invalid_policy
    )
    if harness_events is not None:
        if not result.success:
            harness_events.append(
                HarnessEvent(
                    event_type="invalid_action_proposed",
                    game_type="hanabi",
                    episode_id=episode_id,
                    turn_number=state.turn_number,
                    agent_id=player_id,
                    payload={"action": action.model_dump(), "message": result.message},
                )
            )
            benchmark_penalties = benchmark_penalties if benchmark_penalties is not None else []
            benchmark_penalties.append(
                BenchmarkPenalty(
                    penalty_type="invalid_action",
                    game_type="hanabi",
                    episode_id=episode_id,
                    turn_number=state.turn_number,
                    agent_id=player_id,
                    description=result.message,
                    metadata={"action": action.model_dump()},
                )
            )
            if condition.invalid_action_policy == "human_table":
                harness_events.append(
                    HarnessEvent(
                        event_type="human_table_correction_applied",
                        game_type="hanabi",
                        episode_id=episode_id,
                        turn_number=state.turn_number,
                        agent_id=player_id,
                        payload={"policy": "consume_invalid_turn"},
                    )
                )
                benchmark_penalties.append(
                    BenchmarkPenalty(
                        penalty_type="moderator_correction",
                        game_type="hanabi",
                        episode_id=episode_id,
                        turn_number=state.turn_number,
                        agent_id=player_id,
                        description="Human-table moderator consumed invalid turn and advanced play.",
                        metadata={"policy": "consume_invalid_turn"},
                    )
                )
        if rationale.startswith("Fallback action due to parse failure"):
            harness_events.append(
                HarnessEvent(
                    event_type="fallback_action_used",
                    game_type="hanabi",
                    episode_id=episode_id,
                    turn_number=state.turn_number,
                    agent_id=player_id,
                    payload={"action": action.model_dump(), "rationale": rationale},
                )
            )
            benchmark_penalties = benchmark_penalties if benchmark_penalties is not None else []
            benchmark_penalties.append(
                BenchmarkPenalty(
                    penalty_type="fallback_action",
                    game_type="hanabi",
                    episode_id=episode_id,
                    turn_number=state.turn_number,
                    agent_id=player_id,
                    description="Hanabi fallback action was used after parse failure.",
                    metadata={"action": action.model_dump()},
                )
            )
        harness_events.append(
            HarnessEvent(
                event_type="final_accepted_game_action",
                game_type="hanabi",
                episode_id=episode_id,
                turn_number=turn_log.turn_number,
                agent_id=player_id,
                payload={
                    "action": turn_log.action.model_dump(),
                    "result": turn_log.result.model_dump(),
                },
            )
        )
    
    # Emit turn event if callback provided
    if emit_fn is not None:
        # Include full game state for viewer display
        emit_fn("turn", {
            "turn_number": turn_log.turn_number,
            "player_id": player_id,
            "action": turn_log.action.model_dump(),
            "result": turn_log.result.model_dump(),
            "rationale": rationale,
            "hint_tokens": new_state.hint_tokens,
            "fuse_tokens": new_state.fuse_tokens,
            "score": new_state.score,
            # Full state for viewer (observer sees everything)
            "hands": {
                pid: [{"color": c.color, "number": c.number} for c in hand]
                for pid, hand in new_state.hands.items()
            },
            "knowledge": {
                pid: [k.model_dump() for k in knowledge]
                for pid, knowledge in new_state.knowledge.items()
            },
            "played_cards": dict(new_state.played_cards),
            "discard_pile": [{"color": c.color, "number": c.number} for c in new_state.discard_pile],
            "deck_remaining": len(new_state.deck),
            "current_player": new_state.current_player,
        })
    
    return new_state, turn_log, scratchpad_add


async def run_episode(
    config: HanabiConfig,
    players: list[HanabiPlayerLLM],
    agent_states: AgentStateManager | None = None,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    episode_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    condition: BenchmarkCondition | str | None = None,
    run_manifest: RunManifest | None = None,
) -> HanabiEpisodeRecord:
    """
    Run a complete Hanabi episode.
    
    Args:
        config: Game configuration
        players: List of player agents (must match config.num_players)
        agent_states: Optional state manager (created if not provided)
        emit_fn: Optional callback for emitting events
        episode_id: Optional episode ID (generated if not provided)
        metadata: Optional metadata to include in record
        
    Returns:
        Complete episode record
    """
    if len(players) != config.num_players:
        raise ValueError(f"Expected {config.num_players} players, got {len(players)}")
    
    condition_obj = resolve_condition(condition)
    if agent_states is None and condition_obj.scratchpad_enabled:
        agent_states = AgentStateManager()
    elif not condition_obj.scratchpad_enabled:
        agent_states = None
    
    if episode_id is None:
        episode_id = str(uuid.uuid4())[:8]
    
    # Map player_id to player agent
    player_map = {p.player_id: p for p in players}
    player_ids = [p.player_id for p in players]
    
    # Create game
    state = create_game(config, player_ids)
    seed = config.seed if config.seed is not None else state.config.seed
    
    # Store initial hands for replay
    initial_hands = {
        pid: [card.model_copy() for card in hand]
        for pid, hand in state.hands.items()
    }
    
    # Emit init event with initial game state
    if emit_fn is not None:
        emit_fn("init", {
            "game_type": "hanabi",
            "config": config.model_dump(),
            "player_order": player_ids,
            "episode_id": episode_id,
            # Initial state for viewer
            "hands": {
                pid: [{"color": c.color, "number": c.number} for c in hand]
                for pid, hand in state.hands.items()
            },
            "played_cards": dict(state.played_cards),
            "discard_pile": [],
            "hint_tokens": state.hint_tokens,
            "fuse_tokens": state.fuse_tokens,
            "deck_remaining": len(state.deck),
        })
    
    # Game loop
    turns: list[TurnLog] = []
    harness_events: list[HarnessEvent] = []
    benchmark_penalties: list[BenchmarkPenalty] = []
    
    while not state.game_over:
        current_player_id = state.current_player
        player = player_map[current_player_id]
        
        state, turn_log, scratchpad_add = await run_turn(
            state,
            player,
            agent_states,
            condition_obj,
            emit_fn,
            episode_id,
            harness_events,
            benchmark_penalties,
        )
        turns.append(turn_log)
        harness_events.append(
            HarnessEvent(
                event_type="turn_completed",
                game_type="hanabi",
                episode_id=episode_id,
                turn_number=turn_log.turn_number,
                agent_id=current_player_id,
                payload={
                    "score": state.score,
                    "hint_tokens": state.hint_tokens,
                    "fuse_tokens": state.fuse_tokens,
                },
            )
        )
        
        # Emit scratchpad event
        if emit_fn is not None and scratchpad_add:
            emit_fn("scratchpad", {
                "agent_id": current_player_id,
                "addition": scratchpad_add,
                "turn": turn_log.turn_number,
            })
        
        # Safety limit
        if len(turns) > 200:
            state.game_over = True
            state.game_over_reason = "turn_limit"
            break
    
    # Extract final scratchpads
    agent_scratchpads = {}
    if agent_states is not None:
        agent_scratchpads = {
            agent_id: agent_state.scratchpad
            for agent_id, agent_state in agent_states.get_all_states().items()
            if agent_state.scratchpad
        }
    harness_events.append(
        HarnessEvent(
            event_type="game_completed",
            game_type="hanabi",
            episode_id=episode_id,
            payload={
                "score": state.score,
                "game_over_reason": state.game_over_reason,
            },
        )
    )
    manifest = run_manifest or build_run_manifest(
        game_type="hanabi",
        condition=condition_obj,
        models={
            "player_models": {
                player.player_id: getattr(player.provider, "model", None)
                for player in players
            }
        },
        provider_parameters={},
        seed_schedule=[seed],
        game_rules_version="hanabi_v1",
    )
    
    # Build episode record
    episode = HanabiEpisodeRecord(
        episode_id=episode_id,
        timestamp=datetime.utcnow(),
        config=config,
        seed=seed,
        initial_hands=initial_hands,
        turns=turns,
        final_score=state.score,
        final_played_cards=dict(state.played_cards),
        game_over_reason=state.game_over_reason or "unknown",
        agent_scratchpads=agent_scratchpads,
        metadata={**(metadata or {}), "condition": condition_obj.model_dump(mode="json")},
        run_manifest=manifest,
        harness_events=harness_events,
        benchmark_penalties=benchmark_penalties,
    )
    
    # Emit done event
    if emit_fn is not None:
        metrics = compute_episode_metrics(episode)
        emit_fn("done", {
            "episode_id": episode_id,
            "final_score": state.score,
            "game_over_reason": state.game_over_reason,
            "total_turns": len(turns),
            "metrics": metrics,
            "agent_scratchpads": agent_scratchpads,
        })
    
    return episode
