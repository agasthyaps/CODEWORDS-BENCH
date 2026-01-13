"""Episode runner for complete Codenames games."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from pydantic import BaseModel, Field

from src.engine import (
    Team, GameConfig, GameState, GameMode, EpisodeRecord, Board,
    create_game, Phase,
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
    metadata: dict[str, Any] = Field(default_factory=dict)

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
) -> ExtendedEpisodeRecord:
    """
    Run a complete Codenames episode.

    Args:
        config: Game configuration
        red_team: Red team's agents
        blue_team: Blue team's agents
        max_turns: Maximum turns before draw (safety limit)
        max_discussion_rounds: Max rounds per discussion phase

    Returns:
        ExtendedEpisodeRecord with full game data
    """
    episode_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()

    # Initialize game
    state = create_game(config=config)
    all_traces: list[TurnTraces] = []

    # Determine if we should skip discussion (SINGLE_GUESSER mode)
    skip_discussion = config.mode == GameMode.SINGLE_GUESSER

    # Game loop
    turn_count = 0
    while state.winner is None and turn_count < max_turns:
        turn_count += 1

        # Get current team's agents
        team = state.current_turn
        team_agents = red_team if team == Team.RED else blue_team

        # Run turn
        state, turn_traces = await run_turn(
            team_agents, state, max_discussion_rounds, skip_discussion
        )
        all_traces.append(turn_traces)

        # Check for game over
        if state.phase == Phase.GAME_OVER:
            break

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
        metadata={
            "red_team": _extract_team_metadata(red_team),
            "blue_team": _extract_team_metadata(blue_team),
            "max_discussion_rounds": max_discussion_rounds,
        },
    )

    return episode


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
    )
