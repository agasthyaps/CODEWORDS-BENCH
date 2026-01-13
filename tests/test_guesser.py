"""Tests for the Guesser agent (M2)."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    Team, GameConfig, create_game, apply_clue, transition_to_guessing,
    Phase, DiscussionMessage,
)
from src.agents import (
    AgentConfig, GuesserAgent, run_discussion,
    parse_discussion_response, parse_guess_response, validate_guesses,
    MockProvider,
)


# ============================================================================
# Guess Response Parsing Tests
# ============================================================================

class TestGuessResponseParsing:
    """Tests for parsing guess responses."""

    def test_standard_format_works(self):
        """Standard format should parse correctly."""
        response = """GUESSES: WAVE, BEACH, FISH
REASONING: These words all connect to OCEAN."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.words == ["WAVE", "BEACH", "FISH"]
        assert "OCEAN" in result.reasoning
        assert not result.is_pass

    def test_handles_brackets(self):
        """Should handle brackets around values."""
        response = """GUESSES: [WAVE, BEACH]
REASONING: Bracketed test."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.words == ["WAVE", "BEACH"]

    def test_handles_pass(self):
        """PASS should return empty list with is_pass=True."""
        response = """GUESSES: PASS
REASONING: Too risky to guess."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.words == []
        assert result.is_pass is True
        assert "risky" in result.reasoning.lower()

    def test_handles_pass_with_brackets(self):
        """Should handle [PASS] format."""
        response = """GUESSES: [PASS]
REASONING: Passing."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.is_pass is True

    def test_strips_whitespace(self):
        """Should strip extra whitespace."""
        response = """GUESSES:   WAVE  ,  BEACH  ,  FISH
REASONING: Whitespace test."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.words == ["WAVE", "BEACH", "FISH"]

    def test_case_insensitive(self):
        """Should handle case variations."""
        response = """guesses: wave, beach
reasoning: Case test."""

        result = parse_guess_response(response)

        assert result is not None
        assert result.words == ["WAVE", "BEACH"]

    def test_missing_guesses_returns_none(self):
        """Missing GUESSES should return None."""
        response = """REASONING: No guesses provided."""

        result = parse_guess_response(response)

        assert result is None


# ============================================================================
# Discussion Response Parsing Tests
# ============================================================================

class TestDiscussionResponseParsing:
    """Tests for parsing discussion responses."""

    def test_normal_discussion(self):
        """Normal discussion without consensus should parse."""
        response = "I think WAVE and BEACH connect to the clue. What do you think?"

        result = parse_discussion_response(response)

        assert result.content == response
        assert not result.consensus
        assert result.top_words is None

    def test_consensus_yes(self):
        """Should detect CONSENSUS: YES."""
        response = """I agree with your analysis.

CONSENSUS: YES
TOP: WAVE, BEACH, FISH"""

        result = parse_discussion_response(response)

        assert result.consensus is True
        assert result.top_words == ["WAVE", "BEACH", "FISH"]

    def test_consensus_no(self):
        """CONSENSUS: NO should not trigger consensus."""
        response = """I'm not sure yet.

CONSENSUS: NO"""

        result = parse_discussion_response(response)

        assert not result.consensus

    def test_consensus_case_insensitive(self):
        """Should be case insensitive."""
        response = """consensus: yes
top: wave, beach"""

        result = parse_discussion_response(response)

        assert result.consensus is True
        assert result.top_words == ["WAVE", "BEACH"]


# ============================================================================
# Guess Validation Tests
# ============================================================================

class TestGuessValidation:
    """Tests for guess validation."""

    @pytest.fixture
    def game_state(self):
        """Create a game state in guessing phase."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 3)
        return transition_to_guessing(state)

    def test_truncates_at_invalid_word(self, game_state):
        """Should truncate at word not on board."""
        board_words = [w.upper() for w in game_state.board.words[:2]]
        guesses = board_words + ["NOTAWORD"] + ["ANOTHERVALID"]

        result = validate_guesses(guesses, game_state, max_guesses=10)

        assert result == board_words

    def test_truncates_at_revealed_word(self, game_state):
        """Should truncate at already revealed word."""
        board_words = game_state.board.words[:3]

        # Mark first word as revealed
        game_state.revealed[board_words[0]] = game_state.board.key_by_word[board_words[0]]

        guesses = [board_words[1], board_words[0], board_words[2]]

        result = validate_guesses(guesses, game_state, max_guesses=10)

        # Should truncate at the revealed word (second in list)
        assert result == [board_words[1].upper()]

    def test_removes_duplicates_skip_not_truncate(self, game_state):
        """Should skip duplicates without truncating."""
        board_words = game_state.board.words[:3]

        guesses = [board_words[0], board_words[0], board_words[1], board_words[2]]

        result = validate_guesses(guesses, game_state, max_guesses=10)

        # Duplicates skipped, list continues
        assert result == [w.upper() for w in board_words]

    def test_respects_max_guesses(self, game_state):
        """Should respect max guesses limit."""
        board_words = game_state.board.words[:5]

        result = validate_guesses(board_words, game_state, max_guesses=2)

        assert len(result) == 2
        assert result == [board_words[0].upper(), board_words[1].upper()]

    def test_empty_guesses_returns_empty(self, game_state):
        """Empty guesses should return empty list."""
        result = validate_guesses([], game_state, max_guesses=10)

        assert result == []


# ============================================================================
# Discussion Flow Tests
# ============================================================================

class TestDiscussionFlow:
    """Tests for discussion flow."""

    @pytest.fixture
    def game_state(self):
        """Create a game state in discussion phase."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)
        return state  # Discussion phase

    @pytest.fixture
    def guessers(self):
        """Create two guesser agents."""
        config1 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )
        config2 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_2",
            temperature=0.7,
        )

        provider1 = MockProvider(responses=[
            "I think WAVE connects to the clue.",
            "CONSENSUS: YES\nTOP: WAVE, BEACH",
        ])
        provider2 = MockProvider(responses=[
            "I agree, WAVE seems related. What about BEACH?",
            "CONSENSUS: YES\nTOP: WAVE, BEACH",
        ])

        return [
            GuesserAgent(config1, provider1),
            GuesserAgent(config2, provider2),
        ]

    @pytest.mark.asyncio
    async def test_guessers_alternate(self, game_state, guessers):
        """Guessers should alternate in discussion."""
        messages, traces, _ = await run_discussion(guessers, game_state, max_rounds=3)

        # Check alternation
        agent_ids = [m.agent_id for m in messages]
        expected_pattern = ["red_guesser_1", "red_guesser_2"]

        for i, agent_id in enumerate(agent_ids):
            assert agent_id == expected_pattern[i % 2]

    @pytest.mark.asyncio
    async def test_messages_added_to_transcript(self, game_state, guessers):
        """Messages should be added to public transcript."""
        _, _, final_state = await run_discussion(guessers, game_state, max_rounds=3)

        discussion_events = [
            e for e in final_state.public_transcript
            if isinstance(e, DiscussionMessage)
        ]

        assert len(discussion_events) > 0

    @pytest.mark.asyncio
    async def test_consensus_ends_early(self):
        """Two consecutive CONSENSUS: YES should end discussion."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)

        config1 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )
        config2 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_2",
            temperature=0.7,
        )

        # Both agree immediately
        provider1 = MockProvider(responses=[
            "CONSENSUS: YES\nTOP: WAVE",
        ])
        provider2 = MockProvider(responses=[
            "CONSENSUS: YES\nTOP: WAVE",
        ])

        guessers = [
            GuesserAgent(config1, provider1),
            GuesserAgent(config2, provider2),
        ]

        messages, _, _ = await run_discussion(guessers, state, max_rounds=3)

        # Should end after 2 messages (consensus reached)
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_max_rounds_caps_discussion(self):
        """Discussion should end at max rounds."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)

        config1 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )
        config2 = AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_2",
            temperature=0.7,
        )

        # Never reach consensus
        provider1 = MockProvider(responses=[
            "Not sure...",
            "Still thinking...",
            "Hmm...",
        ])
        provider2 = MockProvider(responses=[
            "Me neither...",
            "Let me think...",
            "Hard to say...",
        ])

        guessers = [
            GuesserAgent(config1, provider1),
            GuesserAgent(config2, provider2),
        ]

        messages, _, _ = await run_discussion(guessers, state, max_rounds=2)

        # max_rounds=2 means 4 messages max (2 per guesser)
        assert len(messages) == 4


# ============================================================================
# Guesser Agent Tests
# ============================================================================

class TestGuesserAgent:
    """Tests for GuesserAgent."""

    @pytest.fixture
    def game_state(self):
        """Create a game state in discussion phase."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)
        return state

    @pytest.fixture
    def guesser_config(self):
        """Create a guesser config."""
        return AgentConfig(
            model="mock-model",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_discuss_produces_message(self, game_state, guesser_config):
        """discuss() should produce a DiscussionMessage."""
        provider = MockProvider(responses=[
            "I think WAVE connects to the clue. The word TESTING might relate to experiments."
        ])
        guesser = GuesserAgent(guesser_config, provider)

        message, trace = await guesser.discuss(game_state, [])

        assert isinstance(message, DiscussionMessage)
        assert message.team == Team.RED
        assert message.agent_id == "red_guesser_1"
        assert "WAVE" in message.content

    @pytest.mark.asyncio
    async def test_make_guesses_produces_list(self, game_state, guesser_config):
        """make_guesses() should produce a list of words."""
        # Use actual board words
        board_words = game_state.board.words[:2]

        provider = MockProvider(responses=[
            f"GUESSES: {board_words[0]}, {board_words[1]}\nREASONING: These connect to the clue."
        ])
        guesser = GuesserAgent(guesser_config, provider)

        guesses, trace = await guesser.make_guesses(game_state, [])

        assert len(guesses) == 2
        assert guesses[0] == board_words[0].upper()
        assert guesses[1] == board_words[1].upper()

    @pytest.mark.asyncio
    async def test_make_guesses_handles_pass(self, game_state, guesser_config):
        """make_guesses() should handle PASS."""
        provider = MockProvider(responses=[
            "GUESSES: PASS\nREASONING: Too risky."
        ])
        guesser = GuesserAgent(guesser_config, provider)

        guesses, trace = await guesser.make_guesses(game_state, [])

        assert guesses == []
        assert trace.parsed_result["is_pass"] is True

    @pytest.mark.asyncio
    async def test_make_guesses_truncates_invalid(self, game_state, guesser_config):
        """make_guesses() should truncate at invalid word."""
        board_word = game_state.board.words[0]

        provider = MockProvider(responses=[
            f"GUESSES: {board_word}, NOTAWORD, {game_state.board.words[1]}\nREASONING: Test."
        ])
        guesser = GuesserAgent(guesser_config, provider)

        guesses, trace = await guesser.make_guesses(game_state, [])

        # Should truncate at NOTAWORD
        assert len(guesses) == 1
        assert guesses[0] == board_word.upper()
        assert len(trace.validation_errors) > 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.asyncio
class TestGuesserIntegration:
    """Integration tests with actual LLM."""

    async def test_guesses_relate_to_clue(self):
        """Guesses should relate to the given clue."""
        import os

        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

        from src.agents import create_provider

        provider = create_provider("openrouter", model="anthropic/claude-3.5-sonnet")

        config1 = AgentConfig(
            model="anthropic/claude-3.5-sonnet",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_1",
            temperature=0.7,
        )
        config2 = AgentConfig(
            model="anthropic/claude-3.5-sonnet",
            role="guesser",
            team=Team.RED,
            agent_id="red_guesser_2",
            temperature=0.7,
        )

        guessers = [
            GuesserAgent(config1, provider),
            GuesserAgent(config2, provider),
        ]

        # Create game with specific clue
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "NATURE", 2)

        messages, traces, final_state = await run_discussion(guessers, state, max_rounds=2)

        # Check that discussion happened
        assert len(messages) >= 2

        # Make guesses
        guesses, trace = await guessers[0].make_guesses(final_state, messages)

        # Guesses should be valid board words
        board_words_upper = {w.upper() for w in state.board.words}
        for guess in guesses:
            assert guess in board_words_upper, f"{guess} not on board"

        print(f"Clue: NATURE")
        print(f"Discussion: {[m.content for m in messages]}")
        print(f"Guesses: {guesses}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
