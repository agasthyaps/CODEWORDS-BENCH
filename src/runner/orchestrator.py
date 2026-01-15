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

if TYPE_CHECKING:
    from .teams import TeamAgents


class TurnTraces(BaseModel):
    """All traces from a single turn."""
    turn_number: int
    team: Team
    clue_trace: AgentTrace
    prediction_trace: AgentTrace | None = None
    discussion_traces: list[AgentTrace]
    guess_trace: AgentTrace | None  # None if team passed without guessing


async def run_clue_phase(
    cluer,  # CluerAgent or GhostCluer
    state: GameState,
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

    clue, trace = await cluer.generate_clue(state)

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

    Returns:
        (new_state, discussion_traces)
    """
    if state.phase != Phase.DISCUSSION:
        raise ValueError(f"Cannot run discussion phase in {state.phase}")

    if skip_discussion or len(guessers) < 2:
        # Single guesser mode - skip discussion
        new_state = transition_to_guessing(state)
        return new_state, []

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

    # Get guesses from the guesser
    guesses, trace = await guesser.make_guesses(state, discussion_messages)

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

    Returns:
        (new_state, turn_traces)
    """
    if state.phase != Phase.CLUE:
        raise ValueError(f"Cannot run turn in {state.phase}")

    turn_number = state.turn_number
    team = state.current_turn

    # Clue phase
    state, clue_trace, should_continue = await run_clue_phase(team_agents.cluer, state)

    if not should_continue:
        # Ghost passed - turn is over
        traces = TurnTraces(
            turn_number=turn_number,
            team=team,
            clue_trace=clue_trace,
            prediction_trace=None,
            discussion_traces=[],
            guess_trace=None,
        )
        return state, traces

    # Prediction step (for ToM): cluer predicts teammate guesses before discussion
    prediction_trace: AgentTrace | None = None
    try:
        if hasattr(team_agents.cluer, "predict_guesses") and state.current_clue is not None:
            prediction_trace = await team_agents.cluer.predict_guesses(state, state.current_clue)
    except Exception:
        # Prediction should never crash the game; store nothing on failure.
        prediction_trace = None

    # Discussion phase
    guessers = team_agents.get_guessers()
    state, discussion_traces = await run_discussion_phase(
        guessers, state, max_discussion_rounds, skip_discussion
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
    )

    traces = TurnTraces(
        turn_number=turn_number,
        team=team,
        clue_trace=clue_trace,
        prediction_trace=prediction_trace,
        discussion_traces=discussion_traces,
        guess_trace=guess_trace,
    )

    return state, traces
