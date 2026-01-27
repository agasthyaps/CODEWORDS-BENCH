"""Clue-giver agent for Codenames."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.engine import (
    Team, Clue, GameState, AgentTrace, validate_clue,
    get_visible_state,
)
from src.core.parsing import extract_scratchpad
from .llm import LLMProvider, LLMResponse


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    model: str
    role: str  # "cluer" | "guesser"
    team: Team
    agent_id: str
    temperature: float = 0.7
    max_retries: int = 3


class ParsedClue(BaseModel):
    """Parsed clue from LLM response."""
    word: str
    number: int
    reasoning: str
    predicted_success: float | None = None  # 0.0-1.0 confidence
    predicted_guesses: list[str] | None = None  # Expected guesser picks in order


def load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = Path(__file__).parent / "prompts" / name
    with open(path, "r") as f:
        return f.read()


def parse_clue_response(response: str) -> ParsedClue | None:
    """
    Parse the clue response from the LLM.

    Handles variations in formatting:
    - Case insensitive
    - Brackets around values
    - Trailing punctuation
    - UNLIMITED keyword -> -1
    """
    # Extract CLUE
    clue_match = re.search(
        r"CLUE\s*:\s*\[?\s*([A-Za-z]+)\s*\]?",
        response,
        re.IGNORECASE
    )
    if not clue_match:
        return None

    word = clue_match.group(1).upper().strip()

    # Extract NUMBER
    number_match = re.search(
        r"NUMBER\s*:\s*\[?\s*(UNLIMITED|\d+)\s*\]?",
        response,
        re.IGNORECASE
    )
    if not number_match:
        return None

    number_str = number_match.group(1).upper().strip()
    if number_str == "UNLIMITED":
        number = -1
    else:
        try:
            number = int(number_str)
        except ValueError:
            return None

    # Extract PREDICTED_SUCCESS (optional, 0.0-1.0)
    predicted_success: float | None = None
    success_match = re.search(
        r"PREDICTED_SUCCESS\s*:\s*\[?\s*([\d.]+)\s*\]?",
        response,
        re.IGNORECASE
    )
    if success_match:
        try:
            val = float(success_match.group(1))
            if 0.0 <= val <= 1.0:
                predicted_success = val
        except ValueError:
            pass

    # Extract PREDICTED_GUESSES (optional, comma-separated words)
    predicted_guesses: list[str] | None = None
    guesses_match = re.search(
        r"PREDICTED_GUESSES\s*:\s*\[?\s*([A-Za-z,\s]+)\s*\]?",
        response,
        re.IGNORECASE
    )
    if guesses_match:
        raw_guesses = guesses_match.group(1)
        # Split by comma and clean up
        guesses = [g.strip().upper() for g in raw_guesses.split(",") if g.strip()]
        if guesses:
            predicted_guesses = guesses

    # Extract REASONING (optional, everything after REASONING:)
    reasoning_match = re.search(
        r"REASONING\s*:\s*(.+?)(?:SCRATCHPAD|$)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    return ParsedClue(
        word=word,
        number=number,
        reasoning=reasoning,
        predicted_success=predicted_success,
        predicted_guesses=predicted_guesses,
    )


def format_board_display(words: list[str], revealed: dict[str, Any]) -> str:
    """Format board words for display, marking revealed ones."""
    lines = []
    for i in range(0, len(words), 5):
        row_words = words[i:i+5]
        formatted = []
        for word in row_words:
            if word in revealed:
                formatted.append(f"[{word}]")  # Mark as revealed
            else:
                formatted.append(word)
        lines.append("  ".join(formatted))
    return "\n".join(lines)


def format_transcript(transcript: list[dict], current_turn: int) -> str:
    """Format the game transcript for display."""
    if not transcript:
        return "(No moves yet - this is the first turn)"

    lines = []
    current_turn_num = None

    for event in transcript:
        turn = event.get("turn_number", 0)
        if turn != current_turn_num:
            current_turn_num = turn
            lines.append(f"\n--- Turn {turn} ---")

        event_type = event.get("event_type")
        team = event.get("team", "")

        if event_type == "clue":
            word = event.get("word", "")
            number = event.get("number", 0)
            num_display = "UNLIMITED" if number == -1 else str(number)
            lines.append(f"{team} Clue: {word} ({num_display})")

        elif event_type == "discussion":
            agent_id = event.get("agent_id", "")
            content = event.get("content", "")
            lines.append(f"{team} {agent_id}: {content}")

        elif event_type == "guess":
            word = event.get("word", "")
            result = event.get("result", "")
            correct = event.get("correct", False)
            status = "CORRECT" if correct else f"WRONG ({result})"
            lines.append(f"{team} guessed {word} -> {status}")

        elif event_type == "pass":
            lines.append(f"{team} passed")

    return "\n".join(lines) if lines else "(No moves yet)"


class CluerAgent:
    """Agent that generates clues for Codenames."""

    def __init__(self, config: AgentConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider
        self.system_prompt = load_prompt_template("cluer_system.md")
        self.turn_prompt_template = load_prompt_template("cluer_turn.md")

    def _build_prompt(
        self, 
        visible_state: dict[str, Any],
        scratchpad_content: str = "",
    ) -> tuple[str, str]:
        """Build the system and user prompts from visible state."""
        team = visible_state["team"]
        opponent_team = "BLUE" if team == "RED" else "RED"

        # Format board display
        board_words = visible_state["board_words"]
        revealed = visible_state.get("revealed", {})
        board_display = format_board_display(board_words, revealed)

        # Get key information
        key = visible_state["key"]
        your_words = ", ".join(sorted(key[team.lower()]))
        opponent_words = ", ".join(sorted(key[opponent_team.lower()]))
        neutral_words = ", ".join(sorted(key["neutral"]))
        assassin_word = ", ".join(sorted(key["assassin"]))

        # Get remaining words for your team
        your_remaining = [w for w in key[team.lower()] if w not in revealed]
        remaining_display = ", ".join(sorted(your_remaining))

        # Format transcript
        transcript = visible_state.get("public_transcript", [])
        turn_number = visible_state.get("turn_number", 1)
        transcript_display = format_transcript(transcript, turn_number)

        # Format scratchpad
        scratchpad_display = scratchpad_content if scratchpad_content else "(empty)"

        # Build prompts
        system = self.system_prompt.format(team=team)
        user = self.turn_prompt_template.format(
            board_words_display=board_display,
            team=team,
            opponent_team=opponent_team,
            your_words=your_words,
            opponent_words=opponent_words,
            neutral_words=neutral_words,
            assassin_word=assassin_word,
            remaining_words=remaining_display,
            transcript_display=transcript_display,
            scratchpad_content=scratchpad_display,
        )

        return system, user

    async def generate_clue(
        self,
        state: GameState,
        scratchpad_content: str = "",
    ) -> tuple[Clue, AgentTrace, str | None]:
        """
        Generate a clue for the current game state.

        Args:
            state: Current game state
            scratchpad_content: Agent's current scratchpad (optional)

        Returns:
            Tuple of (Clue, AgentTrace, scratchpad_addition)
            scratchpad_addition is the new content to add to scratchpad (or None)

        Raises:
            RuntimeError if max retries exceeded
        """
        visible_state = get_visible_state(state, f"{self.config.team.value.lower()}_cluer")
        system_prompt, user_prompt = self._build_prompt(visible_state, scratchpad_content)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        validation_errors: list[str] = []
        retry_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency = 0.0
        all_responses: list[str] = []
        scratchpad_addition: str | None = None

        while retry_count <= self.config.max_retries:
            # Call LLM
            response: LLMResponse = await self.provider.complete(
                messages=messages,
                temperature=self.config.temperature,
            )

            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            total_latency += response.latency_ms
            all_responses.append(response.content)

            # Extract scratchpad addition (from any response, even if clue fails)
            scratchpad_addition = extract_scratchpad(response.content)

            # Parse response
            parsed = parse_clue_response(response.content)

            if parsed is None:
                error = "Could not parse clue from response. Please use the exact format: CLUE: [word], NUMBER: [number], REASONING: [text]"
                validation_errors.append(error)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": f"Error: {error}\n\nPlease try again."})
                retry_count += 1
                continue

            # Validate clue against game rules
            is_valid, error = validate_clue(parsed.word, parsed.number, state)

            if not is_valid:
                validation_errors.append(error or "Unknown validation error")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": f"Error: {error}\n\nPlease provide a different clue that follows all the rules."
                })
                retry_count += 1
                continue

            # Success! Build the Clue and Trace
            clue = Clue(
                turn_number=state.turn_number,
                event_index=0,  # Will be set when added to transcript
                team=self.config.team,
                word=parsed.word,
                number=parsed.number,
            )

            trace = AgentTrace(
                agent_id=self.config.agent_id,
                turn_number=state.turn_number,
                prompt_sent=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
                raw_response="\n---\n".join(all_responses),
                parsed_result={
                    "word": parsed.word,
                    "number": parsed.number,
                    "reasoning": parsed.reasoning,
                },
                validation_errors=validation_errors,
                retry_count=retry_count,
                model=self.config.model,
                temperature=self.config.temperature,
                latency_ms=total_latency,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                predicted_success=parsed.predicted_success,
                predicted_targets=parsed.predicted_guesses,
            )

            return clue, trace, scratchpad_addition

        # Max retries exceeded
        trace = AgentTrace(
            agent_id=self.config.agent_id,
            turn_number=state.turn_number,
            prompt_sent=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}",
            raw_response="\n---\n".join(all_responses),
            parsed_result=None,
            validation_errors=validation_errors,
            retry_count=retry_count,
            model=self.config.model,
            temperature=self.config.temperature,
            latency_ms=total_latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        raise RuntimeError(
            f"Failed to generate valid clue after {self.config.max_retries} retries. "
            f"Errors: {validation_errors}"
        )
