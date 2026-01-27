"""Data models for the Hanabi game engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# Card colors and numbers
Color = Literal["red", "yellow", "green", "blue", "white"]
Number = Literal[1, 2, 3, 4, 5]

COLORS: list[Color] = ["red", "yellow", "green", "blue", "white"]
NUMBERS: list[Number] = [1, 2, 3, 4, 5]

# Card distribution: 1s x3, 2s x2, 3s x2, 4s x2, 5s x1 per color = 10 per color, 50 total
CARD_COUNTS: dict[int, int] = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}


class Card(BaseModel):
    """A Hanabi card with color and number."""

    color: Color
    number: Number

    def __str__(self) -> str:
        return f"{self.color[0].upper()}{self.number}"

    def __hash__(self) -> int:
        return hash((self.color, self.number))


class CardKnowledge(BaseModel):
    """What a player knows about one of their own cards from hints received."""

    model_config = {"arbitrary_types_allowed": True}

    known_color: Color | None = None
    known_number: Number | None = None
    possible_colors: set[Color] = Field(default_factory=lambda: set(COLORS))
    possible_numbers: set[Number] = Field(default_factory=lambda: set(NUMBERS))

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization for sets."""
        data = super().model_dump(**kwargs)
        data["possible_colors"] = sorted(list(self.possible_colors))
        data["possible_numbers"] = sorted(list(self.possible_numbers))
        return data


# Action types
class PlayAction(BaseModel):
    """Play a card from hand by position (0-indexed)."""

    action_type: Literal["play"] = "play"
    card_position: int  # 0 to hand_size-1


class DiscardAction(BaseModel):
    """Discard a card from hand by position (0-indexed)."""

    action_type: Literal["discard"] = "discard"
    card_position: int  # 0 to hand_size-1


class HintAction(BaseModel):
    """Give a hint to another player about their cards."""

    action_type: Literal["hint"] = "hint"
    target_player: str  # player_id
    hint_type: Literal["color", "number"]
    hint_value: str | int  # Color string or Number int


Action = PlayAction | DiscardAction | HintAction


class ActionResult(BaseModel):
    """Result of applying an action."""

    success: bool
    message: str
    card_played: Card | None = None  # For play actions
    card_discarded: Card | None = None  # For discard actions
    was_playable: bool | None = None  # For play actions: was the card actually playable?
    positions_touched: list[int] | None = None  # For hint actions: which positions matched


class TurnLog(BaseModel):
    """Log of a single turn."""

    turn_number: int
    player_id: str
    action: PlayAction | DiscardAction | HintAction
    result: ActionResult
    rationale: str = ""
    
    # State snapshot after action
    hint_tokens_after: int
    fuse_tokens_after: int
    score_after: int


class HanabiConfig(BaseModel):
    """Configuration for a Hanabi game."""

    num_players: int = 3
    hand_size: int = 5  # 5 cards for 2-3 players, 4 for 4-5 players
    max_hints: int = 8
    max_fuses: int = 3
    seed: int | None = None


class HanabiState(BaseModel):
    """The current state of a Hanabi game."""

    config: HanabiConfig
    
    # Hands: player_id -> list of cards (hidden from that player)
    hands: dict[str, list[Card]]
    
    # Knowledge: what each player knows about their own cards
    knowledge: dict[str, list[CardKnowledge]]
    
    # Played cards: color -> highest successfully played number (0 if none)
    played_cards: dict[Color, int]
    
    # Discard pile
    discard_pile: list[Card]
    
    # Deck (hidden from all players)
    deck: list[Card]
    
    # Tokens
    hint_tokens: int
    fuse_tokens: int
    
    # Turn tracking
    current_player_idx: int
    player_order: list[str]
    turn_number: int
    
    # Action history
    action_history: list[TurnLog]
    
    # End game tracking (when deck empties, each player gets one more turn)
    final_round_player: str | None = None  # Player who drew the last card
    game_over: bool = False
    game_over_reason: str | None = None

    @property
    def current_player(self) -> str:
        return self.player_order[self.current_player_idx]

    @property
    def score(self) -> int:
        """Current score (sum of highest played cards per color)."""
        return sum(self.played_cards.values())

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization."""
        data = super().model_dump(**kwargs)
        # Convert Card objects
        data["hands"] = {
            pid: [c.model_dump() for c in cards]
            for pid, cards in self.hands.items()
        }
        data["knowledge"] = {
            pid: [k.model_dump() for k in knowledge]
            for pid, knowledge in self.knowledge.items()
        }
        data["discard_pile"] = [c.model_dump() for c in self.discard_pile]
        data["deck"] = [c.model_dump() for c in self.deck]
        data["action_history"] = [t.model_dump() for t in self.action_history]
        return data


class HanabiEpisodeRecord(BaseModel):
    """Complete record of a Hanabi game episode."""

    episode_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: HanabiConfig
    seed: int
    
    # Initial hands (for replay)
    initial_hands: dict[str, list[Card]]
    
    # Turn history
    turns: list[TurnLog]
    
    # Final state
    final_score: int
    final_played_cards: dict[Color, int]
    game_over_reason: str
    
    # Scratchpads
    agent_scratchpads: dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_filename(self) -> str:
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"hanabi_episode_{self.episode_id}_{ts}.json"

    def save(self, directory: str) -> str:
        """Save episode JSON to a directory. Returns the written filepath."""
        from pathlib import Path
        import json

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        fp = d / self.to_filename()
        data = self.model_dump(mode="json")
        data["timestamp"] = self.timestamp.isoformat()
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)
        return str(fp)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom serialization."""
        data = super().model_dump(**kwargs)
        data["initial_hands"] = {
            pid: [c.model_dump() if hasattr(c, "model_dump") else c for c in cards]
            for pid, cards in self.initial_hands.items()
        }
        data["turns"] = [
            t.model_dump() if hasattr(t, "model_dump") else t
            for t in self.turns
        ]
        return data
