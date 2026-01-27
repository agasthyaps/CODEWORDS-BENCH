"""Orchestrator for running Hanabi games."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable

from src.core.state import AgentStateManager

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
    agent_states: AgentStateManager,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
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
    scratchpad = agent_states.get_scratchpad(player_id)
    
    # Get player's action
    action, rationale, scratchpad_add = await player.decide_action(state, scratchpad)
    
    # Update scratchpad
    if scratchpad_add:
        agent_state = agent_states.get_or_create(player_id)
        agent_state.append_to_scratchpad(state.turn_number, scratchpad_add)
    
    # Apply action
    new_state, result, turn_log = apply_action(state, player_id, action, rationale)
    
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
    
    if agent_states is None:
        agent_states = AgentStateManager()
    
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
    
    while not state.game_over:
        current_player_id = state.current_player
        player = player_map[current_player_id]
        
        state, turn_log, scratchpad_add = await run_turn(
            state, player, agent_states, emit_fn
        )
        turns.append(turn_log)
        
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
    agent_scratchpads = {
        agent_id: agent_state.scratchpad
        for agent_id, agent_state in agent_states.get_all_states().items()
        if agent_state.scratchpad
    }
    
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
        metadata=metadata or {},
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
