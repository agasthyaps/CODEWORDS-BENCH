"""Request/response models for the UI API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.engine import GameMode

# Clue generation mode: controls order of clue generation vs ToM prediction
ClueGenerationMode = Literal["standard", "deliberate"]
# For batch runs, "split" runs half in each mode for A/B comparison
BatchClueGenerationMode = Literal["standard", "deliberate", "split"]


class ModelInfo(BaseModel):
    id: str
    short_name: str | None = None
    provider: str = "openrouter"


class TeamRoleConfig(BaseModel):
    cluer: str
    guesser_1: str
    guesser_2: str | None = None


class TeamSelection(BaseModel):
    red: TeamRoleConfig
    blue: TeamRoleConfig


class CodenamesStartRequest(BaseModel):
    team_selection: TeamSelection
    mode: GameMode = GameMode.STANDARD
    seed: int | None = None
    max_discussion_rounds: int = 3
    max_turns: int = 50
    event_delay_ms: int = 0
    clue_generation_mode: ClueGenerationMode = "standard"


class DecryptoStartRequest(BaseModel):
    team_selection: TeamSelection
    seed: int = 0
    max_rounds: int = 8
    max_discussion_turns_per_guesser: int = 2
    event_delay_ms: int = 0
    clue_generation_mode: ClueGenerationMode = "standard"


class BatchStartRequest(BaseModel):
    game_type: Literal["codenames", "decrypto"]
    count: int = Field(ge=1, le=500)
    seed_count: int = Field(default=1, ge=1, le=500)
    pinned: bool = True
    team_selection: TeamSelection | None = None
    model_pool: list[str] | None = None
    codenames_mode: GameMode = GameMode.STANDARD
    seed: int = 0
    max_discussion_rounds: int = 3
    max_turns: int = 50
    max_rounds: int = 8
    max_discussion_turns_per_guesser: int = 2
    event_delay_ms: int = 0
    clue_generation_mode: BatchClueGenerationMode = "standard"


class JobStartResponse(BaseModel):
    job_id: str


class ReplaySummary(BaseModel):
    replay_id: str
    game_type: Literal["codenames", "decrypto"]
    filename: str
    timestamp: str | None = None
