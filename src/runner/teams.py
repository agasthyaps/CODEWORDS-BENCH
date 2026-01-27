"""Team configuration and ghost teams for Codenames."""

from __future__ import annotations

import random
from enum import Enum
from typing import Any

from pydantic import BaseModel

from src.engine import (
    Team, GameState, AgentTrace, Clue, DiscussionMessage,
    get_visible_state, get_all_unrevealed_words, validate_clue,
)
from src.agents import AgentConfig, CluerAgent, GuesserAgent, LLMProvider


class TeamConfig(BaseModel):
    """Configuration for a team's agents."""
    cluer: AgentConfig
    guesser_1: AgentConfig
    guesser_2: AgentConfig | None = None  # None for SINGLE_GUESSER mode


class TeamAgents:
    """Container for a team's agent instances."""

    def __init__(
        self,
        cluer: CluerAgent,
        guesser_1: GuesserAgent,
        guesser_2: GuesserAgent | None = None,
    ):
        self.cluer = cluer
        self.guesser_1 = guesser_1
        self.guesser_2 = guesser_2

    @property
    def primary_guesser(self) -> GuesserAgent:
        """The designated guesser for final guesses."""
        return self.guesser_1

    def get_guessers(self) -> list[GuesserAgent]:
        """Get list of guessers (1 or 2 depending on mode)."""
        if self.guesser_2 is not None:
            return [self.guesser_1, self.guesser_2]
        return [self.guesser_1]

    @classmethod
    def from_config(
        cls,
        config: TeamConfig,
        provider: LLMProvider,
    ) -> "TeamAgents":
        """Create TeamAgents from config and provider."""
        cluer = CluerAgent(config.cluer, provider)
        guesser_1 = GuesserAgent(config.guesser_1, provider)
        guesser_2 = None
        if config.guesser_2 is not None:
            guesser_2 = GuesserAgent(config.guesser_2, provider)

        return cls(cluer, guesser_1, guesser_2)


class GhostMode(str, Enum):
    """Ghost team behavior mode."""
    PASS = "PASS"      # Always passes - solo mode
    RANDOM = "RANDOM"  # Random legal clue and guesses


class GhostCluer:
    """Ghost clue-giver that either passes or gives random clues."""

    def __init__(self, team: Team, mode: GhostMode = GhostMode.PASS):
        self.team = team
        self.mode = mode
        self.config = AgentConfig(
            model="ghost",
            role="cluer",
            team=team,
            agent_id=f"{team.value.lower()}_ghost_cluer",
            temperature=0.0,
        )

    async def generate_clue(
        self,
        state: GameState,
        scratchpad_content: str = "",
    ) -> tuple[Clue | None, AgentTrace, str | None]:
        """
        Generate a clue (or pass).

        PASS mode: Returns None (signals turn should be skipped)
        RANDOM mode: Returns random legal clue
        
        Returns:
            Tuple of (clue, trace, scratchpad_addition)
        """
        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent="[GHOST]",
            raw_response=f"[GHOST {self.mode.value}]",
            parsed_result={"mode": self.mode.value, "is_ghost": True},
            validation_errors=[],
            retry_count=0,
            model="ghost",
            temperature=0.0,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
        )

        if self.mode == GhostMode.PASS:
            # Return None to signal pass
            return None, trace, None

        # RANDOM mode - generate random legal clue
        # Find a word not on board and not a substring issue
        random_words = [
            "GHOST", "PHANTOM", "SPECTER", "SHADOW", "SPIRIT",
            "HAUNT", "WRAITH", "VAPOR", "MIST", "SHADE",
        ]

        for word in random_words:
            is_valid, _ = validate_clue(word, 1, state)
            if is_valid:
                clue = Clue(
                    turn_number=state.turn_number,
                    event_index=0,
                    team=self.team,
                    word=word,
                    number=random.randint(1, 3),
                )
                trace.parsed_result["clue"] = {"word": clue.word, "number": clue.number}
                return clue, trace, None

        # Fallback - shouldn't happen
        return None, trace, None


class GhostGuesser:
    """Ghost guesser that either passes or guesses randomly."""

    def __init__(self, team: Team, agent_id: str, mode: GhostMode = GhostMode.PASS):
        self.team = team
        self.mode = mode
        self.config = AgentConfig(
            model="ghost",
            role="guesser",
            team=team,
            agent_id=agent_id,
            temperature=0.0,
        )

    async def discuss(
        self,
        state: GameState,
        discussion_so_far: list[DiscussionMessage],
        scratchpad_content: str = "",
    ) -> tuple[DiscussionMessage, AgentTrace, str | None]:
        """Generate a discussion message (or minimal message for pass).
        
        Returns:
            Tuple of (message, trace, scratchpad_addition)
        """
        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent="[GHOST]",
            raw_response=f"[GHOST {self.mode.value}]",
            parsed_result={"mode": self.mode.value, "is_ghost": True},
            validation_errors=[],
            retry_count=0,
            model="ghost",
            temperature=0.0,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
        )

        if self.mode == GhostMode.PASS:
            content = "CONSENSUS: YES\nTOP: PASS"
        else:
            # Random mode - pick random words
            unrevealed = get_all_unrevealed_words(state)
            picks = random.sample(unrevealed, min(2, len(unrevealed)))
            content = f"I think {', '.join(picks)} might work.\n\nCONSENSUS: YES\nTOP: {', '.join(picks)}"

        message = DiscussionMessage(
            turn_number=state.turn_number,
            event_index=0,
            team=self.team,
            agent_id=self.config.agent_id,
            content=content,
        )

        return message, trace, None

    async def make_guesses(
        self,
        state: GameState,
        discussion: list[DiscussionMessage],
        scratchpad_content: str = "",
    ) -> tuple[list[str], AgentTrace, str | None]:
        """Make guesses (empty for pass, random for random mode).
        
        Returns:
            Tuple of (guesses, trace, scratchpad_addition)
        """
        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent="[GHOST]",
            raw_response=f"[GHOST {self.mode.value}]",
            parsed_result={"mode": self.mode.value, "is_ghost": True},
            validation_errors=[],
            retry_count=0,
            model="ghost",
            temperature=0.0,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
        )

        if self.mode == GhostMode.PASS:
            return [], trace, None

        # Random mode - guess random unrevealed words
        unrevealed = get_all_unrevealed_words(state)
        max_guesses = min(
            state.guesses_remaining if state.guesses_remaining > 0 else 1,
            len(unrevealed),
            3,  # Cap at 3 for random
        )
        guesses = random.sample(unrevealed, max_guesses)

        trace.parsed_result["guesses"] = guesses
        return guesses, trace, None


class GhostTeam(TeamAgents):
    """A ghost team that either passes or plays randomly."""

    def __init__(self, team: Team, mode: GhostMode = GhostMode.PASS):
        self.team = team
        self.mode = mode
        self._cluer = GhostCluer(team, mode)
        self._guesser_1 = GhostGuesser(team, f"{team.value.lower()}_ghost_guesser_1", mode)
        self._guesser_2 = GhostGuesser(team, f"{team.value.lower()}_ghost_guesser_2", mode)

    @property
    def cluer(self):
        return self._cluer

    @property
    def guesser_1(self):
        return self._guesser_1

    @property
    def guesser_2(self):
        return self._guesser_2

    @property
    def primary_guesser(self):
        return self._guesser_1

    def get_guessers(self) -> list:
        return [self._guesser_1, self._guesser_2]
