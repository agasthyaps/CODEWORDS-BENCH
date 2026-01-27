"""Turn orchestration for Codenames games."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.engine import (
    Team, Phase, GameState, AgentTrace, Clue, Pass,
    apply_clue, transition_to_guessing, process_guess, process_pass,
    add_discussion_message,
)
from src.agents import CluerAgent, GuesserAgent, run_discussion
from src.core.state import AgentStateManager

if TYPE_CHECKING:
    from .teams import TeamAgents


class TurnTraces(BaseModel):
    """All traces from a single turn."""
    turn_number: int
    team: Team
    clue_trace: AgentTrace
    discussion_traces: list[AgentTrace]
    guess_trace: AgentTrace | None  # None if team passed without guessing


async def run_clue_phase(
    cluer,  # CluerAgent or GhostCluer
    state: GameState,
    agent_states: AgentStateManager | None = None,
) -> tuple[GameState, AgentTrace, bool]:
    """
    Run the clue phase for a team.

    1. Cluer generates a clue
    2. Clue is applied to game state
    3. Phase transitions to DISCUSSION

    Returns:
        (new_state, clue_trace, should_continue)
        should_continue is False if ghost passed (skip rest of turn)
    """
    if state.phase != Phase.CLUE:
        raise ValueError(f"Cannot run clue phase in {state.phase}")

    # Get scratchpad content if available
    scratchpad_content = ""
    if agent_states and hasattr(cluer, 'config'):
        scratchpad_content = agent_states.get_scratchpad(cluer.config.agent_id)

    # Generate clue - all cluers now return (clue, trace, scratchpad_addition)
    clue, trace, scratchpad_addition = await cluer.generate_clue(state, scratchpad_content)
    
    # Update scratchpad if we got an addition
    if agent_states and scratchpad_addition and hasattr(cluer, 'config'):
        agent_state = agent_states.get_or_create(cluer.config.agent_id)
        agent_state.append_to_scratchpad(state.turn_number, scratchpad_addition)

    # Ghost cluer can return None to pass
    if clue is None:
        # End turn without clue - switch to other team
        from src.engine import end_turn
        new_state = end_turn(state)
        return new_state, trace, False

    # Apply the clue to the game state
    new_state = apply_clue(state, clue.word, clue.number)

    return new_state, trace, True


async def run_discussion_phase(
    guessers: list[GuesserAgent],
    state: GameState,
    max_rounds: int = 3,
    skip_discussion: bool = False,
    agent_states: AgentStateManager | None = None,
) -> tuple[GameState, list[AgentTrace]]:
    """
    Run the discussion phase for a team.

    1. Guessers alternate discussing
    2. Messages added to transcript
    3. Ends on consensus or max_rounds

    Args:
        guessers: List of guesser agents
        state: Current game state
        max_rounds: Maximum discussion rounds
        skip_discussion: If True, skip discussion entirely (SINGLE_GUESSER mode)
        agent_states: Optional agent state manager for scratchpads

    Returns:
        (new_state, discussion_traces)
    """
    if state.phase != Phase.DISCUSSION:
        raise ValueError(f"Cannot run discussion phase in {state.phase}")

    if skip_discussion or len(guessers) < 2:
        # Single guesser mode - skip discussion
        new_state = transition_to_guessing(state)
        return new_state, []

    # Note: run_discussion doesn't use scratchpads for simplicity
    # Individual discuss() calls could be updated to use them if needed
    messages, traces, new_state = await run_discussion(
        guessers, state, max_rounds
    )

    # Transition to guessing phase
    new_state = transition_to_guessing(new_state)

    return new_state, traces


async def run_guess_phase(
    guesser: GuesserAgent,
    state: GameState,
    discussion_messages: list,
    agent_states: AgentStateManager | None = None,
) -> tuple[GameState, AgentTrace | None]:
    """
    Run the guess phase for a team.

    1. Designated guesser makes guesses
    2. Each guess is processed
    3. Turn ends on wrong guess, exhausted guesses, or pass

    Returns:
        (new_state, guess_trace)
    """
    if state.phase != Phase.GUESS:
        raise ValueError(f"Cannot run guess phase in {state.phase}")

    # Get scratchpad content if available
    scratchpad_content = ""
    if agent_states and hasattr(guesser, 'config'):
        scratchpad_content = agent_states.get_scratchpad(guesser.config.agent_id)

    # Get guesses from the guesser - all guessers now return (guesses, trace, scratchpad_addition)
    guesses, trace, scratchpad_addition = await guesser.make_guesses(
        state, discussion_messages, scratchpad_content
    )
    
    # Update scratchpad if we got an addition
    if agent_states and scratchpad_addition and hasattr(guesser, 'config'):
        agent_state = agent_states.get_or_create(guesser.config.agent_id)
        agent_state.append_to_scratchpad(state.turn_number, scratchpad_addition)

    new_state = state

    if not guesses:
        # PASS or no guesses - end turn
        new_state = process_pass(new_state)
        return new_state, trace

    # Process each guess
    for guess_word in guesses:
        new_state, result, turn_continues = process_guess(guess_word, new_state)

        # Check if game is over or turn ended
        if new_state.winner is not None:
            break
        if not turn_continues:
            break

    # If we've processed all guesses but turn hasn't ended, end it now
    # This happens when all guesses were correct but we ran out of guess list
    if new_state.phase == Phase.GUESS and new_state.winner is None:
        new_state = process_pass(new_state)

    return new_state, trace


async def run_turn(
    team_agents: "TeamAgents",
    state: GameState,
    max_discussion_rounds: int = 3,
    skip_discussion: bool = False,
    agent_states: AgentStateManager | None = None,
) -> tuple[GameState, TurnTraces]:
    """
    Run a complete turn for a team.

    1. Clue phase
    2. Discussion phase (if 2 guessers and not skipped)
    3. Guess phase

    Args:
        team_agents: The team's agents
        state: Current game state
        max_discussion_rounds: Maximum discussion rounds
        skip_discussion: If True, skip discussion (SINGLE_GUESSER mode)
        agent_states: Optional agent state manager for scratchpads

    Returns:
        (new_state, turn_traces)
    """
    if state.phase != Phase.CLUE:
        raise ValueError(f"Cannot run turn in {state.phase}")

    turn_number = state.turn_number
    team = state.current_turn

    # Clue phase
    state, clue_trace, should_continue = await run_clue_phase(
        team_agents.cluer, state, agent_states
    )

    if not should_continue:
        # Ghost passed - turn is over
        traces = TurnTraces(
            turn_number=turn_number,
            team=team,
            clue_trace=clue_trace,
            discussion_traces=[],
            guess_trace=None,
        )
        return state, traces

    # Discussion phase
    guessers = team_agents.get_guessers()
    state, discussion_traces = await run_discussion_phase(
        guessers, state, max_discussion_rounds, skip_discussion, agent_states
    )

    # Get discussion messages for guess phase
    from src.engine import DiscussionMessage
    discussion_messages = [
        e for e in state.public_transcript
        if isinstance(e, DiscussionMessage) and e.turn_number == turn_number
    ]

    # Guess phase
    state, guess_trace = await run_guess_phase(
        team_agents.primary_guesser,
        state,
        discussion_messages,
        agent_states,
    )

    traces = TurnTraces(
        turn_number=turn_number,
        team=team,
        clue_trace=clue_trace,
        discussion_traces=discussion_traces,
        guess_trace=guess_trace,
    )

    return state, traces
