"""Data models for the Codenames game engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, model_validator


class Team(str, Enum):
    """Team enumeration."""
    RED = "RED"
    BLUE = "BLUE"


class CardType(str, Enum):
    """Card type enumeration."""
    RED = "RED"
    BLUE = "BLUE"
    NEUTRAL = "NEUTRAL"
    ASSASSIN = "ASSASSIN"


class Phase(str, Enum):
    """Game phase enumeration."""
    CLUE = "CLUE"
    DISCUSSION = "DISCUSSION"
    GUESS = "GUESS"
    GAME_OVER = "GAME_OVER"


class GameMode(str, Enum):
    """Game mode enumeration."""
    STANDARD = "STANDARD"          # Full rules, assassin active
    NO_ASSASSIN = "NO_ASSASSIN"    # Assassin card treated as neutral
    SINGLE_GUESSER = "SINGLE_GUESSER"  # 1 guesser per team, no discussion


class GameConfig(BaseModel):
    """Configuration for a Codenames game."""
    words_per_board: int = 25
    red_count: int = 9  # First team advantage
    blue_count: int = 8
    neutral_count: int = 7
    assassin_count: int = 1
    starting_team: Team = Team.RED
    allow_unlimited_clue: bool = True  # Support 0 and -1 (unlimited)
    max_clue_number: int = 9
    seed: int | None = None
    mode: GameMode = GameMode.STANDARD

    @classmethod
    def for_mode(cls, mode: GameMode, seed: int | None = None) -> "GameConfig":
        """Create a GameConfig for the specified mode."""
        if mode == GameMode.NO_ASSASSIN:
            # 9/8/8/0 distribution - no assassin, extra neutral
            return cls(
                red_count=9,
                blue_count=8,
                neutral_count=8,
                assassin_count=0,
                mode=mode,
                seed=seed,
            )
        elif mode == GameMode.SINGLE_GUESSER:
            # Standard distribution, but mode flag for runner
            return cls(mode=mode, seed=seed)
        else:
            # STANDARD mode
            return cls(mode=mode, seed=seed)


class Board(BaseModel):
    """The game board with words and key."""
    words: list[str] = Field(description="25 words in fixed order")
    key_by_category: dict[str, set[str]] = Field(
        description="Mapping of category (red/blue/neutral/assassin) to word sets"
    )
    key_by_word: dict[str, CardType] = Field(
        description="Mapping of word to CardType for O(1) lookup"
    )

    @field_serializer("key_by_category")
    def serialize_key_by_category(self, value: dict[str, set[str]]) -> dict[str, list[str]]:
        """Convert sets to sorted lists for JSON serialization."""
        return {k: sorted(list(v)) for k, v in value.items()}

    @model_validator(mode="before")
    @classmethod
    def convert_lists_to_sets(cls, data: Any) -> Any:
        """Convert lists back to sets when deserializing."""
        if isinstance(data, dict) and "key_by_category" in data:
            data = dict(data)  # Make a copy
            data["key_by_category"] = {
                k: set(v) if isinstance(v, list) else v
                for k, v in data["key_by_category"].items()
            }
        return data


# Transcript Events

class Clue(BaseModel):
    """A clue given by a cluer."""
    event_type: Literal["clue"] = "clue"
    turn_number: int
    event_index: int
    team: Team
    word: str
    number: int  # -1 for unlimited, 0 for zero clue


class Guess(BaseModel):
    """A guess made by a guesser."""
    event_type: Literal["guess"] = "guess"
    turn_number: int
    event_index: int
    team: Team
    word: str
    result: CardType  # What the card actually was
    correct: bool  # Whether it was the team's own card


class Pass(BaseModel):
    """Explicit end-of-guessing by a team."""
    event_type: Literal["pass"] = "pass"
    turn_number: int
    event_index: int
    team: Team


class DiscussionMessage(BaseModel):
    """A message in the discussion phase."""
    event_type: Literal["discussion"] = "discussion"
    turn_number: int
    event_index: int
    team: Team
    agent_id: str
    content: str


# Union type for transcript events
TranscriptEvent = Clue | Guess | Pass | DiscussionMessage


class GameState(BaseModel):
    """The current state of a Codenames game."""
    config: GameConfig
    board: Board
    board_seed: int
    revealed: dict[str, CardType] = Field(default_factory=dict)
    current_turn: Team = Team.RED
    phase: Phase = Phase.CLUE
    turn_number: int = 1
    event_counter: int = 0  # Global, never resets
    current_clue: Clue | None = None
    guesses_remaining: int = 0
    public_transcript: list[TranscriptEvent] = Field(default_factory=list)
    winner: Team | None = None

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization for transcript events."""
        data = super().model_dump(**kwargs)
        # Ensure transcript events are properly serialized
        data["public_transcript"] = [
            event.model_dump() if hasattr(event, "model_dump") else event
            for event in self.public_transcript
        ]
        return data


# Private trace and episode records

class AgentTrace(BaseModel):
    """Private trace of an agent's LLM interaction."""
    agent_id: str
    turn_number: int
    prompt_sent: str
    raw_response: str
    parsed_result: dict[str, Any] | None = None
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    model: str
    temperature: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Prediction tracking for ToM metrics (Cluer only)
    predicted_success: float | None = None  # 0.0-1.0 confidence in guesser success
    predicted_targets: list[str] | None = None  # Expected guesser picks in order


class EpisodeRecord(BaseModel):
    """Complete record of a game episode for benchmark artifacts."""
    episode_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: GameConfig
    board_seed: int
    board: Board
    public_transcript: list[TranscriptEvent]
    agent_traces: list[AgentTrace] = Field(default_factory=list)
    winner: Team | None = None
    final_metrics: dict[str, Any] = Field(default_factory=dict)

    def to_filename(self) -> str:
        """Generate filename for this episode."""
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"episode_{self.episode_id}_{ts}.json"

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization."""
        data = super().model_dump(**kwargs)
        data["timestamp"] = self.timestamp.isoformat()
        data["public_transcript"] = [
            event.model_dump() if hasattr(event, "model_dump") else event
            for event in self.public_transcript
        ]
        return data
