"""Tests for the Clue-giver agent (M1)."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import Team, GameConfig, create_game
from src.agents import (
    AgentConfig, CluerAgent, parse_clue_response,
    MockProvider, create_provider,
)


# ============================================================================
# Parsing Tests
# ============================================================================

class TestClueResponseParsing:
    """Tests for parsing clue responses."""

    def test_standard_format_works(self):
        """Standard format should parse correctly."""
        response = """CLUE: OCEAN
NUMBER: 3
REASONING: Connects WAVE, FISH, and BEACH."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "OCEAN"
        assert result.number == 3
        assert "WAVE" in result.reasoning

    def test_case_insensitive(self):
        """Parsing should be case insensitive."""
        response = """clue: ocean
number: 2
reasoning: Testing case insensitivity."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "OCEAN"
        assert result.number == 2

    def test_strips_punctuation_and_brackets(self):
        """Should handle brackets and extra whitespace."""
        response = """CLUE: [RIVER]
NUMBER: [2]
REASONING: With brackets."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "RIVER"
        assert result.number == 2

    def test_handles_unlimited(self):
        """UNLIMITED should be parsed as -1."""
        response = """CLUE: WATER
NUMBER: UNLIMITED
REASONING: Going for unlimited."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "WATER"
        assert result.number == -1

    def test_handles_zero(self):
        """Zero clue should be parsed correctly."""
        response = """CLUE: DANGER
NUMBER: 0
REASONING: Zero clue to warn about assassin."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "DANGER"
        assert result.number == 0

    def test_missing_clue_returns_none(self):
        """Missing CLUE should return None."""
        response = """NUMBER: 2
REASONING: Missing the clue."""

        result = parse_clue_response(response)

        assert result is None

    def test_missing_number_returns_none(self):
        """Missing NUMBER should return None."""
        response = """CLUE: OCEAN
REASONING: Missing the number."""

        result = parse_clue_response(response)

        assert result is None

    def test_handles_multiline_reasoning(self):
        """Should handle multi-line reasoning."""
        response = """CLUE: FOREST
NUMBER: 2
REASONING: This is my reasoning.
It spans multiple lines.
And includes various thoughts."""

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "FOREST"
        assert "multiple lines" in result.reasoning

    def test_handles_extra_whitespace(self):
        """Should handle extra whitespace."""
        response = """CLUE:    MOUNTAIN
NUMBER:   4
REASONING:    Whitespace test   """

        result = parse_clue_response(response)

        assert result is not None
        assert result.word == "MOUNTAIN"
        assert result.number == 4


# ============================================================================
# Agent Tests (with Mock Provider)
# ============================================================================

class TestCluerAgent:
    """Tests for CluerAgent with mock provider."""

    @pytest.fixture
    def game_state(self):
        """Create a game state for testing."""
        return create_game(config=GameConfig(seed=42))

    @pytest.fixture
    def agent_config(self):
        """Create an agent config."""
        return AgentConfig(
            model="mock-model",
            role="cluer",
            team=Team.RED,
            agent_id="red_cluer",
            temperature=0.7,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_produces_legal_clue(self, game_state, agent_config):
        """Agent should produce a legal clue."""
        provider = MockProvider(responses=[
            "CLUE: UMBRELLA\nNUMBER: 2\nREASONING: Test clue."
        ])
        agent = CluerAgent(agent_config, provider)

        clue, trace = await agent.generate_clue(game_state)

        assert clue.word == "UMBRELLA"
        assert clue.number == 2
        assert clue.team == Team.RED
        assert trace.retry_count == 0
        assert len(trace.validation_errors) == 0

    @pytest.mark.asyncio
    async def test_retries_on_invalid_clue(self, game_state, agent_config):
        """Agent should retry when given invalid clue."""
        # First response uses a board word, second is valid
        board_word = game_state.board.words[0]
        provider = MockProvider(responses=[
            f"CLUE: {board_word}\nNUMBER: 2\nREASONING: Bad clue.",
            "CLUE: UMBRELLA\nNUMBER: 3\nREASONING: Good clue.",
        ])
        agent = CluerAgent(agent_config, provider)

        clue, trace = await agent.generate_clue(game_state)

        assert clue.word == "UMBRELLA"
        assert trace.retry_count == 1
        assert len(trace.validation_errors) == 1
        assert "board" in trace.validation_errors[0].lower()

    @pytest.mark.asyncio
    async def test_retries_on_parse_failure(self, game_state, agent_config):
        """Agent should retry when response can't be parsed."""
        provider = MockProvider(responses=[
            "I think the clue should be OCEAN for 2 words.",  # Invalid format
            "CLUE: UMBRELLA\nNUMBER: 2\nREASONING: Now correct.",
        ])
        agent = CluerAgent(agent_config, provider)

        clue, trace = await agent.generate_clue(game_state)

        assert clue.word == "UMBRELLA"
        assert trace.retry_count == 1
        assert "parse" in trace.validation_errors[0].lower()

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, game_state, agent_config):
        """Agent should raise error after max retries."""
        # All responses are invalid
        board_word = game_state.board.words[0]
        provider = MockProvider(responses=[
            f"CLUE: {board_word}\nNUMBER: 2\nREASONING: Bad.",
            f"CLUE: {board_word}\nNUMBER: 2\nREASONING: Still bad.",
            f"CLUE: {board_word}\nNUMBER: 2\nREASONING: Always bad.",
            f"CLUE: {board_word}\nNUMBER: 2\nREASONING: Never good.",
        ])
        agent = CluerAgent(agent_config, provider)

        with pytest.raises(RuntimeError) as exc_info:
            await agent.generate_clue(game_state)

        assert "retries" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_trace_captures_full_interaction(self, game_state, agent_config):
        """Trace should capture all interaction details."""
        provider = MockProvider(responses=[
            "CLUE: TESTING\nNUMBER: 2\nREASONING: My private reasoning."
        ])
        agent = CluerAgent(agent_config, provider)

        clue, trace = await agent.generate_clue(game_state)

        assert trace.agent_id == "red_cluer"
        assert trace.turn_number == game_state.turn_number
        assert "SYSTEM:" in trace.prompt_sent
        assert "USER:" in trace.prompt_sent
        assert trace.raw_response == "CLUE: TESTING\nNUMBER: 2\nREASONING: My private reasoning."
        assert trace.parsed_result is not None
        assert trace.parsed_result["word"] == "TESTING"
        assert trace.parsed_result["reasoning"] == "My private reasoning."
        assert trace.model == "mock-model"
        assert trace.latency_ms > 0

    @pytest.mark.asyncio
    async def test_clue_word_normalized_to_uppercase(self, game_state, agent_config):
        """Clue word should be normalized to uppercase."""
        provider = MockProvider(responses=[
            "CLUE: umbrella\nNUMBER: 2\nREASONING: Lowercase test."
        ])
        agent = CluerAgent(agent_config, provider)

        clue, trace = await agent.generate_clue(game_state)

        assert clue.word == "UMBRELLA"

    @pytest.mark.asyncio
    async def test_visible_state_includes_key(self, game_state, agent_config):
        """The prompt should include the key for the cluer."""
        provider = MockProvider(responses=[
            "CLUE: TESTING\nNUMBER: 1\nREASONING: Test."
        ])
        agent = CluerAgent(agent_config, provider)

        await agent.generate_clue(game_state)

        # Check that the provider received messages containing key info
        last_messages = provider.last_messages
        user_message = next(m["content"] for m in last_messages if m["role"] == "user")

        # Should contain team words
        assert "Your team's words" in user_message or "RED" in user_message
        assert "Opponent's words" in user_message or "BLUE" in user_message
        assert "ASSASSIN" in user_message


# ============================================================================
# Integration Tests (with real LLM - marked slow)
# ============================================================================

@pytest.mark.slow
@pytest.mark.asyncio
class TestCluerIntegration:
    """Integration tests with actual LLM. Run with: pytest -m slow"""

    async def test_generates_legal_clues_for_10_boards(self):
        """Agent should generate legal clues for 10 different boards."""
        import os

        # Skip if no API key
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        provider = create_provider("openrouter", model="anthropic/claude-3.5-sonnet")
        config = AgentConfig(
            model="anthropic/claude-3.5-sonnet",
            role="cluer",
            team=Team.RED,
            agent_id="red_cluer",
            temperature=0.7,
            max_retries=3,
        )
        agent = CluerAgent(config, provider)

        for seed in range(10):
            game_state = create_game(config=GameConfig(seed=seed))
            clue, trace = await agent.generate_clue(game_state)

            # Verify clue is legal
            from src.engine import validate_clue
            is_valid, error = validate_clue(clue.word, clue.number, game_state)
            assert is_valid, f"Seed {seed}: Invalid clue {clue.word} - {error}"

            # Basic sanity checks
            assert clue.word.isalpha()
            assert clue.word.isupper()
            assert clue.number >= -1

    async def test_clues_relate_to_targets(self):
        """Clues should be semantically related to target words (qualitative)."""
        import os

        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        provider = create_provider("openrouter", model="anthropic/claude-3.5-sonnet")
        config = AgentConfig(
            model="anthropic/claude-3.5-sonnet",
            role="cluer",
            team=Team.RED,
            agent_id="red_cluer",
            temperature=0.7,
            max_retries=3,
        )
        agent = CluerAgent(config, provider)

        # Use a fixed seed for reproducibility
        game_state = create_game(config=GameConfig(seed=123))
        clue, trace = await agent.generate_clue(game_state)

        # The clue should have reasoning that mentions target words
        assert trace.parsed_result is not None
        reasoning = trace.parsed_result.get("reasoning", "")

        # This is a qualitative check - reasoning should exist
        assert len(reasoning) > 10, "Reasoning should be substantial"
        print(f"Clue: {clue.word} {clue.number}")
        print(f"Reasoning: {reasoning}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
