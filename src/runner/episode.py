"""Episode runner for complete Codenames games."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.engine import (
    Team, GameConfig, GameState, GameMode, EpisodeRecord, Board,
    create_game, Phase,
)
from src.core.state import AgentStateManager
from src.core.benchmarking import (
    BenchmarkCondition,
    BenchmarkPenalty,
    HarnessEvent,
    RunManifest,
    build_run_manifest,
    default_benchmark_condition,
    resolve_condition,
    sha256_text,
)
from .orchestrator import run_turn, TurnTraces
from .teams import TeamAgents


class ExtendedEpisodeRecord(BaseModel):
    """Extended episode record with TurnTraces organized by turn."""
    episode_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: GameConfig
    board_seed: int
    board: Board
    public_transcript: list[Any]  # TranscriptEvents
    turn_traces: list[TurnTraces] = Field(default_factory=list)
    winner: Team | None = None
    total_turns: int = 0
    agent_scratchpads: dict[str, str] = Field(default_factory=dict)  # Final scratchpad contents
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_manifest: RunManifest | None = None
    harness_events: list[HarnessEvent] = Field(default_factory=list)
    benchmark_penalties: list[BenchmarkPenalty] = Field(default_factory=list)

    def to_filename(self) -> str:
        """Generate filename for this episode."""
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"episode_{self.episode_id}_{ts}.json"

    def save(self, directory: Path | str) -> Path:
        """Save episode to JSON file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / self.to_filename()

        data = self.model_dump(mode="json")
        # Convert datetime
        data["timestamp"] = self.timestamp.isoformat()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    @classmethod
    def load(cls, filepath: Path | str) -> "ExtendedEpisodeRecord":
        """Load episode from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Parse datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls.model_validate(data)


async def run_episode(
    config: GameConfig,
    red_team: TeamAgents,
    blue_team: TeamAgents,
    max_turns: int = 50,
    max_discussion_rounds: int = 3,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
    condition: BenchmarkCondition | str | None = None,
    run_manifest: RunManifest | None = None,
) -> ExtendedEpisodeRecord:
    """
    Run a complete Codenames episode.

    Args:
        config: Game configuration
        red_team: Red team's agents
        blue_team: Blue team's agents
        max_turns: Maximum turns before draw (safety limit)
        max_discussion_rounds: Max rounds per discussion phase
        emit_fn: Optional callback for emitting events (event_type, data)

    Returns:
        ExtendedEpisodeRecord with full game data
    """
    episode_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()
    condition_obj = resolve_condition(condition)

    def emit(event_type: str, data: dict[str, Any]) -> None:
        if emit_fn:
            emit_fn(event_type, data)

    # Initialize game
    state = create_game(config=config)
    all_traces: list[TurnTraces] = []
    harness_events: list[HarnessEvent] = []
    benchmark_penalties: list[BenchmarkPenalty] = []

    # Initialize agent state manager for scratchpads
    agent_states = AgentStateManager() if condition_obj.scratchpad_enabled else None

    # Determine if we should skip discussion (SINGLE_GUESSER mode)
    skip_discussion = config.mode == GameMode.SINGLE_GUESSER

    emit("game_start", {"episode_id": episode_id, "mode": config.mode.value})

    # Game loop
    turn_count = 0
    while state.winner is None and turn_count < max_turns:
        turn_count += 1

        # Get current team's agents
        team = state.current_turn
        team_agents = red_team if team == Team.RED else blue_team

        emit("turn_start", {"turn": turn_count, "team": team.value})

        # Run turn with agent state manager
        state, turn_traces = await run_turn(
            team_agents, state, max_discussion_rounds, skip_discussion, agent_states
        )
        all_traces.append(turn_traces)
        _record_turn_harness_events(
            episode_id=episode_id,
            turn_traces=turn_traces,
            harness_events=harness_events,
            benchmark_penalties=benchmark_penalties,
        )

        # Emit turn complete - extract clue info from trace if available
        clue_info = None
        if turn_traces.clue_trace and turn_traces.clue_trace.parsed_result:
            pr = turn_traces.clue_trace.parsed_result
            clue_info = {
                "word": pr.get("clue_word") or pr.get("word"),
                "count": pr.get("clue_count") or pr.get("count"),
            }

        emit("turn_complete", {
            "turn": turn_count,
            "team": team.value,
            "clue": clue_info,
        })
        harness_events.append(
            HarnessEvent(
                event_type="turn_completed",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=turn_count,
                team=team.value,
                payload={"clue": clue_info},
            )
        )

        # Check for game over
        if state.phase == Phase.GAME_OVER:
            break

    # Extract final scratchpad contents
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
            game_type="codenames",
            episode_id=episode_id,
            payload={"winner": state.winner.value if state.winner else None},
        )
    )

    manifest = run_manifest or build_run_manifest(
        game_type="codenames",
        condition=condition_obj,
        models={
            "red_team": _extract_team_metadata(red_team),
            "blue_team": _extract_team_metadata(blue_team),
        },
        provider_parameters={
            "max_discussion_rounds": max_discussion_rounds,
            "max_turns": max_turns,
        },
        seed_schedule=[state.board_seed],
        game_rules_version=config.mode.value,
    )

    # Build episode record
    episode = ExtendedEpisodeRecord(
        episode_id=episode_id,
        timestamp=start_time,
        config=config,
        board_seed=state.board_seed,
        board=state.board,
        public_transcript=[e.model_dump() for e in state.public_transcript],
        turn_traces=all_traces,
        winner=state.winner,
        total_turns=turn_count,
        agent_scratchpads=agent_scratchpads,
        metadata={
            "red_team": _extract_team_metadata(red_team),
            "blue_team": _extract_team_metadata(blue_team),
            "max_discussion_rounds": max_discussion_rounds,
            "condition": condition_obj.model_dump(mode="json"),
        },
        run_manifest=manifest,
        harness_events=harness_events,
        benchmark_penalties=benchmark_penalties,
    )

    return episode


def _trace_role(agent_id: str) -> str:
    if "cluer" in agent_id:
        return "cluer"
    if "guesser" in agent_id:
        return "guesser"
    return "agent"


def _trace_team(agent_id: str) -> str | None:
    if agent_id.startswith("red_"):
        return "RED"
    if agent_id.startswith("blue_"):
        return "BLUE"
    return None


def _record_trace_events(
    *,
    episode_id: str,
    trace: Any | None,
    harness_events: list[HarnessEvent],
    benchmark_penalties: list[BenchmarkPenalty],
) -> None:
    if trace is None:
        return

    role = _trace_role(trace.agent_id)
    team = _trace_team(trace.agent_id)
    prompt = trace.prompt_sent or ""
    response = trace.raw_response or ""
    payload_base = {
        "model": trace.model,
        "temperature": trace.temperature,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
    }
    harness_events.append(
        HarnessEvent(
            event_type="model_prompt_sent",
            game_type="codenames",
            episode_id=episode_id,
            turn_number=trace.turn_number,
            team=team,
            agent_id=trace.agent_id,
            role=role,
            payload={
                **payload_base,
                "prompt_hash": sha256_text(prompt),
                "prompt_chars": len(prompt),
            },
        )
    )
    harness_events.append(
        HarnessEvent(
            event_type="raw_model_response_received",
            game_type="codenames",
            episode_id=episode_id,
            turn_number=trace.turn_number,
            team=team,
            agent_id=trace.agent_id,
            role=role,
            payload={
                **payload_base,
                "response_hash": sha256_text(response),
                "response_chars": len(response),
                "latency_ms": trace.latency_ms,
            },
        )
    )

    parse_failed = any("parse" in e.lower() or "could not" in e.lower() for e in trace.validation_errors)
    harness_events.append(
        HarnessEvent(
            event_type="parse_failed" if parse_failed else "parse_succeeded",
            game_type="codenames",
            episode_id=episode_id,
            turn_number=trace.turn_number,
            team=team,
            agent_id=trace.agent_id,
            role=role,
            payload={"validation_errors": trace.validation_errors},
        )
    )
    if parse_failed:
        benchmark_penalties.append(
            BenchmarkPenalty(
                penalty_type="parse_failure",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=trace.turn_number,
                team=team,
                agent_id=trace.agent_id,
                description="Model response failed parsing or validation.",
                metadata={"validation_errors": trace.validation_errors},
            )
        )

    if trace.retry_count:
        harness_events.append(
            HarnessEvent(
                event_type="repair_prompt_issued",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=trace.turn_number,
                team=team,
                agent_id=trace.agent_id,
                role=role,
                payload={"retry_count": trace.retry_count},
            )
        )
        benchmark_penalties.append(
            BenchmarkPenalty(
                penalty_type="repair_prompt",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=trace.turn_number,
                team=team,
                agent_id=trace.agent_id,
                points=float(trace.retry_count),
                description="Neutral repair prompt was issued before accepting an action.",
            )
        )

    parsed = trace.parsed_result or {}
    fallback = parsed.get("fallback") or any("fallback" in e.lower() for e in trace.validation_errors)
    if fallback:
        harness_events.append(
            HarnessEvent(
                event_type="fallback_action_used",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=trace.turn_number,
                team=team,
                agent_id=trace.agent_id,
                role=role,
                payload={"parsed_result": parsed},
            )
        )
        benchmark_penalties.append(
            BenchmarkPenalty(
                penalty_type="fallback_action",
                game_type="codenames",
                episode_id=episode_id,
                turn_number=trace.turn_number,
                team=team,
                agent_id=trace.agent_id,
                description="Harness fallback action was used.",
                metadata={"parsed_result": parsed},
            )
        )


def _record_turn_harness_events(
    *,
    episode_id: str,
    turn_traces: TurnTraces,
    harness_events: list[HarnessEvent],
    benchmark_penalties: list[BenchmarkPenalty],
) -> None:
    _record_trace_events(
        episode_id=episode_id,
        trace=turn_traces.clue_trace,
        harness_events=harness_events,
        benchmark_penalties=benchmark_penalties,
    )
    for trace in turn_traces.discussion_traces:
        _record_trace_events(
            episode_id=episode_id,
            trace=trace,
            harness_events=harness_events,
            benchmark_penalties=benchmark_penalties,
        )
    _record_trace_events(
        episode_id=episode_id,
        trace=turn_traces.guess_trace,
        harness_events=harness_events,
        benchmark_penalties=benchmark_penalties,
    )
    harness_events.append(
        HarnessEvent(
            event_type="final_accepted_game_action",
            game_type="codenames",
            episode_id=episode_id,
            turn_number=turn_traces.turn_number,
            team=turn_traces.team.value,
            payload={
                "clue": (
                    turn_traces.clue_trace.parsed_result
                    if turn_traces.clue_trace is not None
                    else None
                ),
                "guess": (
                    turn_traces.guess_trace.parsed_result
                    if turn_traces.guess_trace is not None
                    else None
                ),
            },
        )
    )


def _extract_team_metadata(team: TeamAgents) -> dict[str, Any]:
    """Extract metadata about a team's agents."""
    from .teams import GhostTeam

    if isinstance(team, GhostTeam):
        return {
            "type": "ghost",
            "mode": team.mode.value,
        }

    metadata = {
        "type": "llm",
        "cluer_model": team.cluer.config.model,
        "guesser_1_model": team.guesser_1.config.model,
    }

    if team.guesser_2 is not None:
        metadata["guesser_2_model"] = team.guesser_2.config.model

    return metadata


async def run_single_team_episode(
    config: GameConfig,
    real_team: TeamAgents,
    real_team_color: Team = Team.RED,
    max_turns: int = 50,
    max_discussion_rounds: int = 3,
    emit_fn: Callable[[str, dict[str, Any]], None] | None = None,
) -> ExtendedEpisodeRecord:
    """
    Run a single-team episode against a PASS ghost.

    The ghost team always passes, so the real team just needs to
    clear all their words to win.

    Args:
        config: Game configuration
        real_team: The real team's agents
        real_team_color: Which color the real team plays
        max_turns: Maximum turns before draw
        max_discussion_rounds: Max rounds per discussion
        emit_fn: Optional callback for emitting events

    Returns:
        ExtendedEpisodeRecord
    """
    from .teams import GhostTeam, GhostMode

    ghost_color = Team.BLUE if real_team_color == Team.RED else Team.RED
    ghost_team = GhostTeam(ghost_color, GhostMode.PASS)

    if real_team_color == Team.RED:
        red_team = real_team
        blue_team = ghost_team
    else:
        red_team = ghost_team
        blue_team = real_team

    return await run_episode(
        config=config,
        red_team=red_team,
        blue_team=blue_team,
        max_turns=max_turns,
        max_discussion_rounds=max_discussion_rounds,
        emit_fn=emit_fn,
    )
