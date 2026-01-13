"""Guesser agent for Codenames."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.engine import (
    Team, GameState, AgentTrace, DiscussionMessage,
    get_visible_state, add_discussion_message,
)
from .llm import LLMProvider
from .cluer import AgentConfig, format_board_display, format_transcript


class ParsedGuesses(BaseModel):
    """Parsed guesses from LLM response."""
    words: list[str]
    reasoning: str
    is_pass: bool = False


class ParsedDiscussion(BaseModel):
    """Parsed discussion message from LLM response."""
    content: str
    consensus: bool = False
    top_words: list[str] | None = None


def load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = Path(__file__).parent / "prompts" / name
    with open(path, "r") as f:
        return f.read()


def parse_discussion_response(response: str) -> ParsedDiscussion:
    """
    Parse a discussion response, checking for consensus signal.

    Looks for:
    - CONSENSUS: YES (case insensitive)
    - TOP: word1, word2, word3
    """
    content = response.strip()

    # Check for consensus
    consensus_match = re.search(
        r"CONSENSUS\s*:\s*(YES|NO)",
        response,
        re.IGNORECASE
    )
    consensus = bool(consensus_match and consensus_match.group(1).upper() == "YES")

    # Extract TOP words if present
    top_words = None
    if consensus:
        top_match = re.search(
            r"TOP\s*:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE
        )
        if top_match:
            words_str = top_match.group(1)
            # Split by comma and clean up
            top_words = [
                w.strip().upper()
                for w in words_str.split(",")
                if w.strip()
            ]

    return ParsedDiscussion(
        content=content,
        consensus=consensus,
        top_words=top_words,
    )


def parse_guess_response(response: str) -> ParsedGuesses | None:
    """
    Parse the guesses from LLM response.

    Handles:
    - GUESSES: word1, word2, word3
    - GUESSES: PASS
    - Brackets around values
    """
    # Check for PASS
    pass_match = re.search(
        r"GUESSES\s*:\s*\[?\s*PASS\s*\]?",
        response,
        re.IGNORECASE
    )
    if pass_match:
        reasoning_match = re.search(
            r"REASONING\s*:\s*(.+)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return ParsedGuesses(words=[], reasoning=reasoning, is_pass=True)

    # Extract guesses
    guesses_match = re.search(
        r"GUESSES\s*:\s*\[?\s*(.+?)\s*\]?\s*(?:\n|REASONING|$)",
        response,
        re.IGNORECASE
    )
    if not guesses_match:
        return None

    words_str = guesses_match.group(1)
    words = [
        w.strip().upper()
        for w in words_str.split(",")
        if w.strip() and w.strip().upper() != "REASONING"
    ]

    # Extract reasoning
    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return ParsedGuesses(words=words, reasoning=reasoning, is_pass=False)


def validate_guesses(
    guesses: list[str],
    state: GameState,
    max_guesses: int
) -> list[str]:
    """
    Validate and truncate guesses.

    - Must be on board (case-insensitive)
    - Must not be already revealed
    - No duplicates (skip, don't truncate)
    - Truncate at first invalid word
    - Respect max guesses
    """
    board_words_upper = {w.upper() for w in state.board.words}
    revealed_upper = {w.upper() for w in state.revealed}

    validated: list[str] = []
    seen: set[str] = set()

    for word in guesses:
        word = word.upper()

        # Check for duplicates - skip (don't truncate)
        if word in seen:
            continue
        seen.add(word)

        # Check if on board
        if word not in board_words_upper:
            break  # Truncate at invalid

        # Check if already revealed
        if word in revealed_upper:
            break  # Truncate at invalid

        validated.append(word)

        # Check max guesses
        if len(validated) >= max_guesses:
            break

    return validated


def format_discussion_display(
    discussion: list[DiscussionMessage | dict]
) -> str:
    """Format the current discussion for display."""
    if not discussion:
        return "(No discussion yet - you are first to speak)"

    lines = []
    for msg in discussion:
        if isinstance(msg, DiscussionMessage):
            agent_id = msg.agent_id
            content = msg.content
        else:
            agent_id = msg.get("agent_id", "Unknown")
            content = msg.get("content", "")

        lines.append(f"**{agent_id}:** {content}")

    return "\n\n".join(lines)


class GuesserAgent:
    """Agent that discusses and guesses for Codenames."""

    def __init__(self, config: AgentConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider

        # Load prompt templates
        self.discussion_system = load_prompt_template("guesser_discussion.md")
        self.discussion_turn = load_prompt_template("guesser_discussion_turn.md")
        self.guess_system = load_prompt_template("guesser_guess.md")
        self.guess_turn = load_prompt_template("guesser_guess_turn.md")

    def _get_clue_info(self, state: GameState) -> tuple[str, int, int]:
        """Get current clue word, number, and max guesses."""
        if state.current_clue is None:
            raise ValueError("No current clue in state")

        word = state.current_clue.word
        number = state.current_clue.number

        # Calculate max guesses
        if number == -1:  # Unlimited
            max_guesses = 25
        elif number == 0:
            max_guesses = 1
        else:
            max_guesses = number + 1

        return word, number, max_guesses

    def _build_discussion_prompt(
        self,
        state: GameState,
        discussion_so_far: list[DiscussionMessage],
    ) -> tuple[str, str]:
        """Build prompts for discussion phase."""
        visible = get_visible_state(
            state,
            f"{self.config.team.value.lower()}_guesser_1"
        )

        team = visible["team"]
        clue_word, clue_number, max_guesses = self._get_clue_info(state)

        # Format board (unrevealed only)
        unrevealed = [
            w for w in visible["board_words"]
            if w not in visible.get("revealed", {})
        ]
        board_display = "  ".join(unrevealed)

        # Format transcript (history before current turn)
        transcript = [
            e for e in visible["public_transcript"]
            if e.get("turn_number", 0) < state.turn_number
        ]
        transcript_display = format_transcript(transcript, state.turn_number)

        # Format current discussion
        discussion_display = format_discussion_display(discussion_so_far)

        # Number display
        num_display = "UNLIMITED" if clue_number == -1 else str(clue_number)

        system = self.discussion_system.format(team=team)
        user = self.discussion_turn.format(
            board_words_display=board_display,
            team=team,
            clue_word=clue_word,
            clue_number=num_display,
            max_guesses=max_guesses,
            transcript_display=transcript_display,
            discussion_display=discussion_display,
        )

        return system, user

    def _build_guess_prompt(
        self,
        state: GameState,
        discussion: list[DiscussionMessage],
    ) -> tuple[str, str]:
        """Build prompts for final guess phase."""
        visible = get_visible_state(
            state,
            f"{self.config.team.value.lower()}_guesser_1"
        )

        team = visible["team"]
        clue_word, clue_number, max_guesses = self._get_clue_info(state)

        # Format board (unrevealed only)
        unrevealed = [
            w for w in visible["board_words"]
            if w not in visible.get("revealed", {})
        ]
        board_display = "  ".join(unrevealed)

        # Format discussion
        discussion_display = format_discussion_display(discussion)

        # Number display
        num_display = "UNLIMITED" if clue_number == -1 else str(clue_number)

        system = self.guess_system.format(
            team=team,
            max_guesses=max_guesses,
        )
        user = self.guess_turn.format(
            board_words_display=board_display,
            team=team,
            clue_word=clue_word,
            clue_number=num_display,
            max_guesses=max_guesses,
            discussion_display=discussion_display,
        )

        return system, user

    async def discuss(
        self,
        state: GameState,
        discussion_so_far: list[DiscussionMessage],
    ) -> tuple[DiscussionMessage, AgentTrace]:
        """
        Add one message to the discussion.

        No retry - returns whatever the LLM produces.
        Checks for consensus signal.
        """
        system_prompt, user_prompt = self._build_discussion_prompt(
            state, discussion_so_far
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.provider.complete(
            messages=messages,
            temperature=self.config.temperature,
        )

        # Parse response
        parsed = parse_discussion_response(response.content)

        # Create discussion message
        message = DiscussionMessage(
            turn_number=state.turn_number,
            event_index=0,  # Will be set when added to transcript
            team=self.config.team,
            agent_id=self.config.agent_id,
            content=parsed.content,
        )

        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
            raw_response=response.content,
            parsed_result={
                "content": parsed.content,
                "consensus": parsed.consensus,
                "top_words": parsed.top_words,
            },
            validation_errors=[],
            retry_count=0,
            model=self.config.model,
            temperature=self.config.temperature,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return message, trace

    async def make_guesses(
        self,
        state: GameState,
        discussion: list[DiscussionMessage],
    ) -> tuple[list[str], AgentTrace]:
        """
        Produce final ordered guesses.

        Validates and truncates guesses (no retry).
        Returns empty list for PASS.
        """
        system_prompt, user_prompt = self._build_guess_prompt(state, discussion)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.provider.complete(
            messages=messages,
            temperature=self.config.temperature,
        )

        # Parse response
        parsed = parse_guess_response(response.content)

        validation_errors = []

        if parsed is None:
            # Couldn't parse - return empty (like a pass)
            validation_errors.append("Could not parse guesses from response")
            guesses: list[str] = []
        elif parsed.is_pass:
            guesses = []
        else:
            # Validate and truncate
            _, _, max_guesses = self._get_clue_info(state)
            guesses = validate_guesses(parsed.words, state, max_guesses)

            if len(guesses) < len(parsed.words):
                validation_errors.append(
                    f"Truncated guesses from {len(parsed.words)} to {len(guesses)}"
                )

        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
            raw_response=response.content,
            parsed_result={
                "words": guesses,
                "reasoning": parsed.reasoning if parsed else "",
                "is_pass": parsed.is_pass if parsed else True,
                "raw_words": parsed.words if parsed else [],
            },
            validation_errors=validation_errors,
            retry_count=0,
            model=self.config.model,
            temperature=self.config.temperature,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return guesses, trace


async def run_discussion(
    guessers: list[GuesserAgent],
    state: GameState,
    max_rounds: int = 3,
) -> tuple[list[DiscussionMessage], list[AgentTrace], GameState]:
    """
    Run the discussion phase between guessers.

    Guessers alternate (guesser_1, guesser_2, guesser_1, ...).
    Each message is added to public_transcript immediately.
    Ends when: 2 consecutive CONSENSUS: YES signals, OR max_rounds reached.

    Returns:
        Tuple of (messages, traces, updated_state)
    """
    if len(guessers) != 2:
        raise ValueError("Exactly 2 guessers required")

    messages: list[DiscussionMessage] = []
    traces: list[AgentTrace] = []
    current_state = state

    # Track consecutive consensus signals
    consecutive_consensus = 0
    max_messages = max_rounds * 2  # 2 guessers per round

    for i in range(max_messages):
        guesser = guessers[i % 2]

        # Generate discussion message
        message, trace = await guesser.discuss(current_state, messages)
        traces.append(trace)

        # Add message to state transcript
        current_state = add_discussion_message(
            current_state,
            message.agent_id,
            message.content,
        )

        # Get the message with proper event_index from the updated state
        actual_message = current_state.public_transcript[-1]
        if isinstance(actual_message, DiscussionMessage):
            messages.append(actual_message)
        else:
            messages.append(message)

        # Check for consensus
        parsed = parse_discussion_response(message.content)
        if parsed.consensus:
            consecutive_consensus += 1
            if consecutive_consensus >= 2:
                # Both agreed - end discussion
                break
        else:
            consecutive_consensus = 0

    return messages, traces, current_state
