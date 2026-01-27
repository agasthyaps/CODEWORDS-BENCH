"""Request/response models for the UI API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.engine import GameMode


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


class DecryptoStartRequest(BaseModel):
    team_selection: TeamSelection
    seed: int | None = None
    max_rounds: int = 8
    max_discussion_turns_per_guesser: int = 2
    event_delay_ms: int = 0


class BatchStartRequest(BaseModel):
    """
    Simplified batch request with explicit seed control.
    
    Seed modes:
    - "random": Generate `count` unique random seeds
    - "fixed": Use single `fixed_seed` for all `count` games
    - "list": Run exactly the seeds in `seed_list`
    
    Game types:
    - "codenames": Run only Codenames games
    - "decrypto": Run only Decrypto games
    - "hanabi": Run only Hanabi games
    - "both": Run both Codenames and Decrypto for each seed (comparative analysis)
    """
    game_type: Literal["codenames", "decrypto", "hanabi", "both"]
    team_selection: TeamSelection | None = None  # For codenames/decrypto
    
    # Seed configuration
    seed_mode: Literal["random", "fixed", "list"] = "random"
    count: int = Field(default=5, ge=1, le=100)  # for random/fixed modes
    fixed_seed: int | None = None  # for fixed mode
    seed_list: list[int] | None = None  # for list mode
    
    # Codenames-specific options
    codenames_mode: GameMode = GameMode.STANDARD
    max_discussion_rounds: int = 3
    max_turns: int = 50
    
    # Decrypto-specific options  
    max_rounds: int = 8
    max_discussion_turns_per_guesser: int = 2
    
    # Hanabi-specific options
    player_models: list[str] | None = None  # 3 model IDs for Hanabi players


class JobStartResponse(BaseModel):
    job_id: str


class HanabiStartRequest(BaseModel):
    """Request to start a Hanabi game with 3 players."""
    player_models: list[str]  # 3 model IDs for the players
    seed: int | None = None
    event_delay_ms: int = 0


class ReplaySummary(BaseModel):
    replay_id: str
    game_type: Literal["codenames", "decrypto", "hanabi"]
    filename: str
    timestamp: str | None = None
