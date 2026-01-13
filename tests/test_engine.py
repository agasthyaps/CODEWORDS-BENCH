"""Tests for the Codenames game engine (M0)."""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import (
    Team, CardType, Phase, GameConfig, Board, GameState,
    Clue, Guess, Pass, DiscussionMessage, EpisodeRecord,
    generate_board, validate_clue, process_guess, process_pass,
    check_winner, get_visible_state, add_event, create_game, apply_clue,
    transition_to_guessing, add_discussion_message, load_wordlist,
)


# ============================================================================
# Board Generation Tests
# ============================================================================

class TestBoardGeneration:
    """Tests for board generation."""

    def test_generates_25_unique_words(self):
        """Board should have exactly 25 unique words."""
        board, _ = generate_board()
        assert len(board.words) == 25
        assert len(set(board.words)) == 25

    def test_correct_distribution(self):
        """Board should have correct card type distribution (9/8/7/1)."""
        board, _ = generate_board()

        assert len(board.key_by_category["red"]) == 9
        assert len(board.key_by_category["blue"]) == 8
        assert len(board.key_by_category["neutral"]) == 7
        assert len(board.key_by_category["assassin"]) == 1

    def test_seeding_produces_identical_boards(self):
        """Same seed should produce identical boards."""
        config = GameConfig(seed=42)

        board1, seed1 = generate_board(config=config)
        board2, seed2 = generate_board(config=config)

        assert seed1 == seed2 == 42
        assert board1.words == board2.words
        assert board1.key_by_category == board2.key_by_category

    def test_different_seeds_produce_different_boards(self):
        """Different seeds should produce different boards."""
        board1, _ = generate_board(config=GameConfig(seed=1))
        board2, _ = generate_board(config=GameConfig(seed=2))

        # Extremely unlikely to be the same
        assert board1.words != board2.words

    def test_both_key_representations_consistent(self):
        """key_by_category and key_by_word should be consistent."""
        board, _ = generate_board()

        # Check all words in key_by_word are in exactly one category
        for word in board.words:
            assert word in board.key_by_word
            card_type = board.key_by_word[word]
            category = card_type.value.lower()
            assert word in board.key_by_category[category]

        # Check all category words are in key_by_word
        for category, words in board.key_by_category.items():
            expected_type = CardType(category.upper())
            for word in words:
                assert board.key_by_word[word] == expected_type

    def test_custom_word_list(self):
        """Should work with custom word list."""
        custom_words = [f"WORD{i}" for i in range(100)]
        board, _ = generate_board(word_list=custom_words)

        assert len(board.words) == 25
        assert all(word in custom_words for word in board.words)


# ============================================================================
# Clue Validation Tests
# ============================================================================

class TestClueValidation:
    """Tests for clue validation."""

    @pytest.fixture
    def game_state(self):
        """Create a game state for testing."""
        config = GameConfig(seed=42)
        return create_game(config=config)

    def test_rejects_exact_board_word(self, game_state):
        """Clue that matches a board word exactly should be rejected."""
        board_word = game_state.board.words[0]
        valid, error = validate_clue(board_word, 2, game_state)

        assert not valid
        assert "word on the board" in error.lower()

    def test_rejects_case_variations(self, game_state):
        """Clue matching board word with different case should be rejected."""
        board_word = game_state.board.words[0].lower()
        valid, error = validate_clue(board_word, 2, game_state)

        assert not valid
        assert "word on the board" in error.lower()

    def test_rejects_substring_clue_in_board_word(self, game_state):
        """Clue that is substring of board word should be rejected."""
        # Find a word we can make a substring of
        for word in game_state.board.words:
            if len(word) > 3:
                substring = word[:3]
                # Make sure substring isn't itself on the board
                if substring not in [w.upper() for w in game_state.board.words]:
                    valid, error = validate_clue(substring, 2, game_state)
                    assert not valid
                    assert "substring" in error.lower()
                    return
        pytest.skip("No suitable word found for substring test")

    def test_rejects_board_word_substring_of_clue(self, game_state):
        """Clue containing board word as substring should be rejected."""
        board_word = game_state.board.words[0]
        clue_with_board_word = board_word + "ING"
        valid, error = validate_clue(clue_with_board_word, 2, game_state)

        assert not valid
        assert "substring" in error.lower()

    def test_rejects_previous_clues(self, game_state):
        """Clue already used should be rejected."""
        # Apply a clue first
        state = apply_clue(game_state, "TESTING", 2)

        # Try to use same clue again (transition to next turn first)
        state = transition_to_guessing(state)
        state, _, _ = process_guess(state.board.words[0], state)
        # Now we're in clue phase for opposite team

        valid, error = validate_clue("TESTING", 1, state)
        assert not valid
        assert "already used" in error.lower()

    def test_rejects_special_characters(self, game_state):
        """Clue with special characters should be rejected."""
        valid, error = validate_clue("HELLO-WORLD", 2, game_state)
        assert not valid
        assert "non-alphabetic" in error.lower()

        valid, error = validate_clue("TEST123", 2, game_state)
        assert not valid
        assert "non-alphabetic" in error.lower()

    def test_accepts_valid_clue(self, game_state):
        """Valid clue should be accepted."""
        valid, error = validate_clue("UMBRELLA", 2, game_state)
        assert valid
        assert error is None

    def test_handles_zero_clue(self, game_state):
        """Zero clue should be valid."""
        valid, error = validate_clue("TESTING", 0, game_state)
        assert valid
        assert error is None

    def test_handles_unlimited_clue(self, game_state):
        """Unlimited clue (-1) should be valid when allowed."""
        valid, error = validate_clue("TESTING", -1, game_state)
        assert valid
        assert error is None

    def test_rejects_unlimited_when_disabled(self):
        """Unlimited clue should be rejected when not allowed."""
        config = GameConfig(seed=42, allow_unlimited_clue=False)
        state = create_game(config=config)

        valid, error = validate_clue("TESTING", -1, state)
        assert not valid
        assert "not allowed" in error.lower()

    def test_rejects_number_exceeding_max(self, game_state):
        """Number exceeding max should be rejected."""
        valid, error = validate_clue("TESTING", 15, game_state)
        assert not valid
        assert "exceeds maximum" in error.lower()


# ============================================================================
# Guess Processing Tests
# ============================================================================

class TestGuessProcessing:
    """Tests for guess processing."""

    @pytest.fixture
    def guessing_state(self):
        """Create a game state in guessing phase."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 3)
        state = transition_to_guessing(state)
        return state

    def test_correct_guess_reveals_and_continues(self, guessing_state):
        """Correct guess should reveal card and allow continuation."""
        # Find a word belonging to current team
        team_words = guessing_state.board.key_by_category[
            guessing_state.current_turn.value.lower()
        ]
        target_word = list(team_words)[0]

        new_state, result, continues = process_guess(target_word, guessing_state)

        assert target_word in new_state.revealed
        assert result == CardType(guessing_state.current_turn.value)
        assert continues is True

    def test_decrements_guesses_remaining(self, guessing_state):
        """Correct guess should decrement guesses remaining."""
        initial_guesses = guessing_state.guesses_remaining

        team_words = list(guessing_state.board.key_by_category[
            guessing_state.current_turn.value.lower()
        ])
        new_state, _, _ = process_guess(team_words[0], guessing_state)

        assert new_state.guesses_remaining == initial_guesses - 1

    def test_neutral_ends_turn(self, guessing_state):
        """Guessing neutral word should end turn."""
        neutral_word = list(guessing_state.board.key_by_category["neutral"])[0]

        new_state, result, continues = process_guess(neutral_word, guessing_state)

        assert result == CardType.NEUTRAL
        assert continues is False
        assert new_state.current_turn != guessing_state.current_turn

    def test_opponent_word_ends_turn(self, guessing_state):
        """Guessing opponent word should end turn."""
        opponent_team = "blue" if guessing_state.current_turn == Team.RED else "red"
        opponent_word = list(guessing_state.board.key_by_category[opponent_team])[0]

        new_state, result, continues = process_guess(opponent_word, guessing_state)

        assert result == CardType(opponent_team.upper())
        assert continues is False
        assert new_state.current_turn != guessing_state.current_turn

    def test_opponent_guess_can_trigger_their_win(self):
        """Guessing opponent's last word should make them win."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", -1)  # Unlimited guesses
        state = transition_to_guessing(state)

        # Reveal all but one blue word
        opponent_team = "blue" if state.current_turn == Team.RED else "red"
        opponent_words = list(state.board.key_by_category[opponent_team])

        for word in opponent_words[:-1]:
            state.revealed[word] = CardType(opponent_team.upper())

        # Guess the last opponent word
        last_word = opponent_words[-1]
        new_state, result, continues = process_guess(last_word, state)

        assert new_state.winner == Team(opponent_team.upper())
        assert new_state.phase == Phase.GAME_OVER

    def test_assassin_ends_game(self, guessing_state):
        """Guessing assassin should end game with opposing team winning."""
        assassin_word = list(guessing_state.board.key_by_category["assassin"])[0]

        new_state, result, continues = process_guess(assassin_word, guessing_state)

        assert result == CardType.ASSASSIN
        assert continues is False
        assert new_state.phase == Phase.GAME_OVER
        assert new_state.winner != guessing_state.current_turn

    def test_invalid_guess_not_on_board(self, guessing_state):
        """Guess not on board should end turn silently."""
        new_state, result, continues = process_guess("NOTAWORD", guessing_state)

        assert result is None
        assert continues is False
        assert new_state.current_turn != guessing_state.current_turn
        assert "NOTAWORD" not in new_state.revealed

    def test_already_revealed_guess(self, guessing_state):
        """Guessing already revealed word should end turn silently."""
        # Reveal a word first
        team_words = list(guessing_state.board.key_by_category[
            guessing_state.current_turn.value.lower()
        ])
        first_word = team_words[0]
        state_after_first, _, _ = process_guess(first_word, guessing_state)

        # Try guessing same word again
        new_state, result, continues = process_guess(first_word, state_after_first)

        assert result is None
        assert continues is False
        assert new_state.current_turn != state_after_first.current_turn

    def test_exhausted_guesses_ends_turn(self):
        """Running out of guesses should end turn."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 1)  # 2 guesses total
        state = transition_to_guessing(state)

        team_words = list(state.board.key_by_category[
            state.current_turn.value.lower()
        ])

        # First guess (correct)
        state, _, continues = process_guess(team_words[0], state)
        assert continues is True

        # Second guess (correct, but exhausts guesses)
        state, _, continues = process_guess(team_words[1], state)
        assert continues is False
        assert state.current_turn != Team.RED  # Turn ended


# ============================================================================
# Pass Tests
# ============================================================================

class TestPass:
    """Tests for pass functionality."""

    @pytest.fixture
    def guessing_state(self):
        """Create a game state in guessing phase."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 3)
        state = transition_to_guessing(state)
        return state

    def test_pass_ends_turn(self, guessing_state):
        """Pass should end the turn."""
        new_state = process_pass(guessing_state)

        assert new_state.current_turn != guessing_state.current_turn
        assert new_state.phase == Phase.CLUE

    def test_pass_recorded_in_transcript(self, guessing_state):
        """Pass should be recorded in transcript."""
        new_state = process_pass(guessing_state)

        pass_events = [e for e in new_state.public_transcript if isinstance(e, Pass)]
        assert len(pass_events) == 1
        assert pass_events[0].team == guessing_state.current_turn


# ============================================================================
# Win Condition Tests
# ============================================================================

class TestWinConditions:
    """Tests for win condition checking."""

    def test_all_words_revealed_wins(self):
        """Team with all words revealed should win."""
        state = create_game(config=GameConfig(seed=42))

        # Reveal all red words
        for word in state.board.key_by_category["red"]:
            state.revealed[word] = CardType.RED

        winner = check_winner(state)
        assert winner == Team.RED

    def test_opponent_hits_assassin_wins(self):
        """Opponent hitting assassin should result in win."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 3)
        state = transition_to_guessing(state)

        assassin_word = list(state.board.key_by_category["assassin"])[0]
        state, _, _ = process_guess(assassin_word, state)

        # Red was guessing, so Blue wins
        assert state.winner == Team.BLUE

    def test_no_winner_mid_game(self):
        """Should be no winner in middle of game."""
        state = create_game(config=GameConfig(seed=42))

        # Reveal some but not all words
        red_words = list(state.board.key_by_category["red"])
        for word in red_words[:5]:
            state.revealed[word] = CardType.RED

        winner = check_winner(state)
        assert winner is None


# ============================================================================
# Visibility Tests
# ============================================================================

class TestVisibility:
    """Tests for state visibility."""

    @pytest.fixture
    def game_state(self):
        """Create a game state with some history."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)
        return state

    def test_cluer_sees_key(self, game_state):
        """Cluer should see the key."""
        visible = get_visible_state(game_state, "red_cluer")

        assert "key" in visible
        assert "red" in visible["key"]
        assert "blue" in visible["key"]
        assert "neutral" in visible["key"]
        assert "assassin" in visible["key"]

    def test_guesser_has_no_key_field(self, game_state):
        """Guesser should not have key field (not null - absent)."""
        visible = get_visible_state(game_state, "red_guesser_1")

        assert "key" not in visible

    def test_everyone_sees_full_transcript(self, game_state):
        """All roles should see full public transcript."""
        roles = [
            "red_cluer", "blue_cluer",
            "red_guesser_1", "red_guesser_2",
            "blue_guesser_1", "blue_guesser_2",
        ]

        for role in roles:
            visible = get_visible_state(game_state, role)
            assert "public_transcript" in visible
            assert len(visible["public_transcript"]) > 0

    def test_everyone_sees_board_words(self, game_state):
        """All roles should see board words."""
        visible = get_visible_state(game_state, "red_guesser_1")

        assert "board_words" in visible
        assert len(visible["board_words"]) == 25

    def test_everyone_sees_revealed_cards(self, game_state):
        """All roles should see revealed cards."""
        # Reveal a card first
        game_state = transition_to_guessing(game_state)
        team_word = list(game_state.board.key_by_category["red"])[0]
        game_state, _, _ = process_guess(team_word, game_state)

        visible = get_visible_state(game_state, "blue_guesser_1")

        assert "revealed" in visible
        assert team_word in visible["revealed"]


# ============================================================================
# Transcript Tests
# ============================================================================

class TestTranscript:
    """Tests for transcript functionality."""

    def test_events_ordered_by_event_index(self):
        """Events should be ordered by global event_index."""
        state = create_game(config=GameConfig(seed=42))

        # Add several events
        state = apply_clue(state, "TESTING", 2)
        state = transition_to_guessing(state)

        team_words = list(state.board.key_by_category["red"])
        state, _, _ = process_guess(team_words[0], state)
        state, _, _ = process_guess(team_words[1], state)

        # Check ordering
        indices = [e.event_index for e in state.public_transcript]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # All unique

    def test_multiple_events_same_turn_distinct_indices(self):
        """Multiple events in same turn should have distinct indices."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 3)
        state = transition_to_guessing(state)

        team_words = list(state.board.key_by_category["red"])
        state, _, _ = process_guess(team_words[0], state)
        state, _, _ = process_guess(team_words[1], state)

        # Get events from turn 1
        turn_1_events = [e for e in state.public_transcript if e.turn_number == 1]
        turn_1_indices = [e.event_index for e in turn_1_events]

        assert len(set(turn_1_indices)) == len(turn_1_indices)

    def test_serialization_roundtrip(self):
        """GameState should survive JSON serialization roundtrip."""
        state = create_game(config=GameConfig(seed=42))
        state = apply_clue(state, "TESTING", 2)
        state = transition_to_guessing(state)

        team_word = list(state.board.key_by_category["red"])[0]
        state, _, _ = process_guess(team_word, state)

        # Serialize
        data = state.model_dump()
        json_str = json.dumps(data)

        # Deserialize
        parsed = json.loads(json_str)
        # Note: Full roundtrip to GameState would need custom parsing
        # for transcript events; here we verify serialization works

        assert parsed["turn_number"] == state.turn_number
        assert len(parsed["public_transcript"]) == len(state.public_transcript)


# ============================================================================
# Full Game (Scripted) Tests
# ============================================================================

class TestFullGame:
    """Tests for complete game flow."""

    def test_scripted_game_to_completion(self):
        """Can play through a game with hardcoded moves."""
        state = create_game(config=GameConfig(seed=42))

        # Get words for each category
        red_words = list(state.board.key_by_category["red"])
        blue_words = list(state.board.key_by_category["blue"])

        red_idx = 0
        blue_idx = 0

        # Clue words that won't conflict with board
        red_clues = ["CRIMSON", "SCARLET", "RUBY", "GARNET", "CHERRY",
                     "RUSSET", "MAROON", "CORAL", "SALMON", "ROSE"]
        blue_clues = ["AZURE", "COBALT", "NAVY", "SAPPHIRE", "INDIGO",
                      "CYAN", "TEAL", "DENIM", "ROYAL", "SKY"]

        # Play until someone wins
        max_turns = 20
        turn = 0

        while state.winner is None and turn < max_turns:
            if state.current_turn == Team.RED:
                # Red's turn
                state = apply_clue(state, red_clues[turn % len(red_clues)], 2)
                state = transition_to_guessing(state)

                for _ in range(3):
                    if red_idx >= len(red_words):
                        break
                    state, result, continues = process_guess(
                        red_words[red_idx], state
                    )
                    red_idx += 1
                    if not continues or state.winner:
                        break

            else:
                # Blue's turn
                state = apply_clue(state, blue_clues[turn % len(blue_clues)], 2)
                state = transition_to_guessing(state)

                for _ in range(3):
                    if blue_idx >= len(blue_words):
                        break
                    state, result, continues = process_guess(
                        blue_words[blue_idx], state
                    )
                    blue_idx += 1
                    if not continues or state.winner:
                        break

            turn += 1

        assert state.winner is not None
        assert state.phase == Phase.GAME_OVER

    def test_no_moves_after_game_over(self):
        """No moves should be accepted after game over."""
        state = create_game(config=GameConfig(seed=42))

        # Trigger assassin immediately
        state = apply_clue(state, "TESTING", 1)
        state = transition_to_guessing(state)

        assassin_word = list(state.board.key_by_category["assassin"])[0]
        state, _, _ = process_guess(assassin_word, state)

        assert state.phase == Phase.GAME_OVER
        assert state.winner is not None

        # Try to make moves
        with pytest.raises(ValueError):
            apply_clue(state, "NEWCLUE", 2)

        with pytest.raises(ValueError):
            process_guess("ANYTHING", state)

        with pytest.raises(ValueError):
            process_pass(state)

    def test_correct_winner_determined(self):
        """Winner should be correctly determined."""
        state = create_game(config=GameConfig(seed=42))

        # Red reveals all their words
        state = apply_clue(state, "WINNING", -1)  # Unlimited guesses
        state = transition_to_guessing(state)

        red_words = list(state.board.key_by_category["red"])
        for word in red_words:
            if state.winner:
                break
            state, _, _ = process_guess(word, state)

        assert state.winner == Team.RED


# ============================================================================
# Episode Record Tests
# ============================================================================

class TestEpisodeRecord:
    """Tests for episode record functionality."""

    def test_episode_filename_format(self):
        """Episode filename should include ID and timestamp."""
        from datetime import datetime

        episode = EpisodeRecord(
            episode_id="test123",
            timestamp=datetime(2024, 1, 15, 10, 30, 45),
            config=GameConfig(seed=42),
            board_seed=42,
            board=generate_board(config=GameConfig(seed=42))[0],
            public_transcript=[],
        )

        filename = episode.to_filename()

        assert "test123" in filename
        assert "20240115" in filename
        assert filename.endswith(".json")

    def test_episode_serialization(self):
        """Episode should serialize to JSON."""
        board, seed = generate_board(config=GameConfig(seed=42))

        episode = EpisodeRecord(
            episode_id="test123",
            config=GameConfig(seed=42),
            board_seed=seed,
            board=board,
            public_transcript=[
                Clue(turn_number=1, event_index=1, team=Team.RED, word="TEST", number=2)
            ],
            winner=Team.RED,
        )

        data = episode.model_dump()
        json_str = json.dumps(data)

        # Should not raise
        parsed = json.loads(json_str)
        assert parsed["episode_id"] == "test123"
        assert parsed["winner"] == "RED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
