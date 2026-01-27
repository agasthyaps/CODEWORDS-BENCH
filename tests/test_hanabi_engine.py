"""Tests for Hanabi game engine logic."""

import pytest
from src.hanabi.models import (
    Card,
    CardKnowledge,
    HanabiConfig,
    HanabiState,
    PlayAction,
    DiscardAction,
    HintAction,
    COLORS,
    NUMBERS,
    CARD_COUNTS,
)
from src.hanabi.game import (
    create_game,
    create_deck,
    apply_action,
    apply_play,
    apply_discard,
    apply_hint,
    check_terminal,
    deal_card,
    is_playable,
    is_critical,
    count_remaining,
)


class TestDeckCreation:
    """Tests for deck creation."""

    def test_deck_has_correct_size(self):
        """Deck should have 50 cards (10 per color)."""
        deck = create_deck(seed=42)
        assert len(deck) == 50

    def test_deck_has_correct_distribution(self):
        """Each color should have correct card counts."""
        deck = create_deck(seed=42)
        
        for color in COLORS:
            color_cards = [c for c in deck if c.color == color]
            assert len(color_cards) == 10
            
            for number, expected_count in CARD_COUNTS.items():
                num_cards = [c for c in color_cards if c.number == number]
                assert len(num_cards) == expected_count, \
                    f"{color} {number}s: expected {expected_count}, got {len(num_cards)}"

    def test_deck_deterministic_with_seed(self):
        """Same seed should produce same deck order."""
        deck1 = create_deck(seed=123)
        deck2 = create_deck(seed=123)
        
        for c1, c2 in zip(deck1, deck2):
            assert c1.color == c2.color
            assert c1.number == c2.number

    def test_different_seeds_different_decks(self):
        """Different seeds should produce different deck orders."""
        deck1 = create_deck(seed=1)
        deck2 = create_deck(seed=2)
        
        # Should be different (extremely unlikely to be same)
        differences = sum(1 for c1, c2 in zip(deck1, deck2) 
                        if c1.color != c2.color or c1.number != c2.number)
        assert differences > 0


class TestGameCreation:
    """Tests for game initialization."""

    def test_creates_correct_number_of_players(self):
        """Should create game with specified number of players."""
        for num_players in [2, 3, 4, 5]:
            config = HanabiConfig(num_players=num_players, seed=42)
            state = create_game(config)
            assert len(state.player_order) == num_players
            assert len(state.hands) == num_players

    def test_hands_have_correct_size(self):
        """Each player should have the correct hand size."""
        config = HanabiConfig(num_players=3, hand_size=5, seed=42)
        state = create_game(config)
        
        for player_id, hand in state.hands.items():
            assert len(hand) == 5

    def test_knowledge_initialized(self):
        """Each player should have initialized card knowledge."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        for player_id in state.player_order:
            knowledge = state.knowledge[player_id]
            assert len(knowledge) == len(state.hands[player_id])
            
            for k in knowledge:
                assert k.known_color is None
                assert k.known_number is None
                assert k.possible_colors == set(COLORS)
                assert k.possible_numbers == set(NUMBERS)

    def test_initial_tokens(self):
        """Game should start with max tokens."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        assert state.hint_tokens == config.max_hints
        assert state.fuse_tokens == config.max_fuses

    def test_initial_played_cards_empty(self):
        """All stacks should start at 0."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        for color in COLORS:
            assert state.played_cards[color] == 0


class TestPlayAction:
    """Tests for play actions."""

    def test_successful_play(self):
        """Playing a playable card should succeed."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Find a 1 in player 1's hand (always playable at start)
        player_hand = state.hands["player_1"]
        one_idx = next((i for i, c in enumerate(player_hand) if c.number == 1), None)
        
        if one_idx is not None:
            action = PlayAction(card_position=one_idx)
            new_state, result, _ = apply_action(state, "player_1", action)
            
            assert result.success
            assert result.was_playable
            assert result.card_played is not None
            assert result.card_played.number == 1
            
            # Stack should be updated
            assert new_state.played_cards[result.card_played.color] == 1

    def test_failed_play_loses_fuse(self):
        """Playing unplayable card should lose a fuse token."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Find a non-1 in player 1's hand
        player_hand = state.hands["player_1"]
        non_one_idx = next((i for i, c in enumerate(player_hand) if c.number != 1), None)
        
        if non_one_idx is not None:
            initial_fuses = state.fuse_tokens
            action = PlayAction(card_position=non_one_idx)
            new_state, result, _ = apply_action(state, "player_1", action)
            
            assert result.success  # Action was valid
            assert result.was_playable == False  # But card wasn't playable
            assert new_state.fuse_tokens == initial_fuses - 1

    def test_play_draws_new_card(self):
        """After playing, player should draw a new card."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        initial_hand_size = len(state.hands["player_1"])
        initial_deck_size = len(state.deck)
        
        action = PlayAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", action)
        
        if result.success:
            # Hand size should be same (played one, drew one)
            assert len(new_state.hands["player_1"]) == initial_hand_size
            # Deck should be one smaller
            assert len(new_state.deck) == initial_deck_size - 1

    def test_completing_stack_gives_hint(self):
        """Playing a 5 should give a bonus hint token (if not at max)."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Set up a nearly complete stack
        state.played_cards["red"] = 4
        state.hint_tokens = 5  # Not at max
        
        # Put a red 5 in player's hand
        state.hands["player_1"][0] = Card(color="red", number=5)
        
        action = PlayAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert result.success
        assert result.was_playable
        assert new_state.played_cards["red"] == 5
        assert new_state.hint_tokens == 6  # Got bonus hint


class TestDiscardAction:
    """Tests for discard actions."""

    def test_discard_gains_hint(self):
        """Discarding should regain a hint token."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        state.hint_tokens = 5  # Below max
        
        action = DiscardAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert result.success
        assert new_state.hint_tokens == 6
        assert result.card_discarded is not None

    def test_cannot_discard_at_max_hints(self):
        """Cannot discard when at max hint tokens."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        assert state.hint_tokens == 8  # Max by default
        
        action = DiscardAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert not result.success
        assert "max" in result.message.lower()

    def test_discard_draws_new_card(self):
        """After discarding, player should draw a new card."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        state.hint_tokens = 5
        
        initial_hand_size = len(state.hands["player_1"])
        
        action = DiscardAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", action)
        
        if result.success:
            assert len(new_state.hands["player_1"]) == initial_hand_size


class TestHintAction:
    """Tests for hint actions."""

    def test_color_hint_updates_knowledge(self):
        """Color hint should update target's knowledge."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Get target's hand to find a valid hint
        target_hand = state.hands["player_2"]
        hint_color = target_hand[0].color
        
        action = HintAction(
            target_player="player_2",
            hint_type="color",
            hint_value=hint_color,
        )
        
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert result.success
        assert result.positions_touched is not None
        assert 0 in result.positions_touched
        
        # Check knowledge updated
        for i, k in enumerate(new_state.knowledge["player_2"]):
            if target_hand[i].color == hint_color:
                assert k.known_color == hint_color
            else:
                assert hint_color not in k.possible_colors

    def test_number_hint_updates_knowledge(self):
        """Number hint should update target's knowledge."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        target_hand = state.hands["player_2"]
        hint_number = target_hand[0].number
        
        action = HintAction(
            target_player="player_2",
            hint_type="number",
            hint_value=hint_number,
        )
        
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert result.success
        assert 0 in result.positions_touched

    def test_hint_costs_token(self):
        """Giving a hint should cost a hint token."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        initial_hints = state.hint_tokens
        
        target_hand = state.hands["player_2"]
        action = HintAction(
            target_player="player_2",
            hint_type="color",
            hint_value=target_hand[0].color,
        )
        
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert result.success
        assert new_state.hint_tokens == initial_hints - 1

    def test_cannot_hint_with_no_tokens(self):
        """Cannot give hint with no hint tokens."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        state.hint_tokens = 0
        
        target_hand = state.hands["player_2"]
        action = HintAction(
            target_player="player_2",
            hint_type="color",
            hint_value=target_hand[0].color,
        )
        
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert not result.success
        assert "token" in result.message.lower()

    def test_cannot_hint_self(self):
        """Cannot give hint to yourself."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        action = HintAction(
            target_player="player_1",  # Same as current player
            hint_type="color",
            hint_value="red",
        )
        
        new_state, result, _ = apply_action(state, "player_1", action)
        
        assert not result.success
        assert "yourself" in result.message.lower()

    def test_hint_must_touch_card(self):
        """Hint must touch at least one card."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Find a color not in target's hand
        target_hand = state.hands["player_2"]
        target_colors = {c.color for c in target_hand}
        missing_color = next((c for c in COLORS if c not in target_colors), None)
        
        if missing_color:
            action = HintAction(
                target_player="player_2",
                hint_type="color",
                hint_value=missing_color,
            )
            
            new_state, result, _ = apply_action(state, "player_1", action)
            
            assert not result.success
            assert "at least one" in result.message.lower()


class TestTerminalConditions:
    """Tests for game ending conditions."""

    def test_fuse_out_ends_game(self):
        """Game ends when all fuse tokens are lost."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        state.fuse_tokens = 0
        
        is_over, reason = check_terminal(state)
        
        assert is_over
        assert reason == "fuse_out"

    def test_perfect_score_ends_game(self):
        """Game ends with perfect score of 25."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Set all stacks to 5
        for color in COLORS:
            state.played_cards[color] = 5
        
        is_over, reason = check_terminal(state)
        
        assert is_over
        assert reason == "perfect_score"
        assert state.score == 25

    def test_game_continues_normally(self):
        """Game should continue if no terminal condition met."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        is_over, reason = check_terminal(state)
        
        assert not is_over
        assert reason is None


class TestTurnAdvancement:
    """Tests for turn advancement."""

    def test_turn_advances_after_action(self):
        """Turn should advance to next player after action."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        assert state.current_player == "player_1"
        
        action = DiscardAction(card_position=0)
        state.hint_tokens = 5  # Allow discard
        new_state, result, _ = apply_action(state, "player_1", action)
        
        if result.success:
            assert new_state.current_player == "player_2"
            assert new_state.turn_number == 2

    def test_turn_wraps_around(self):
        """Turn should wrap from last player to first."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        state.current_player_idx = 2  # player_3's turn
        state.hint_tokens = 5
        
        action = DiscardAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_3", action)
        
        if result.success:
            assert new_state.current_player == "player_1"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_playable(self):
        """Test playability check."""
        played = {"red": 2, "blue": 0, "green": 5, "yellow": 3, "white": 1}
        
        assert is_playable(Card(color="red", number=3), played)
        assert not is_playable(Card(color="red", number=2), played)
        assert not is_playable(Card(color="red", number=4), played)
        assert is_playable(Card(color="blue", number=1), played)
        assert not is_playable(Card(color="green", number=1), played)  # Stack complete

    def test_count_remaining(self):
        """Test remaining card count."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # Initially all cards available
        assert count_remaining(state, Card(color="red", number=1)) == 3
        assert count_remaining(state, Card(color="red", number=5)) == 1
        
        # After discard
        state.discard_pile.append(Card(color="red", number=1))
        assert count_remaining(state, Card(color="red", number=1)) == 2

    def test_is_critical(self):
        """Test critical card detection."""
        config = HanabiConfig(num_players=3, seed=42)
        state = create_game(config)
        
        # 5s are always critical (only 1 copy)
        assert is_critical(state, Card(color="red", number=5))
        
        # After discarding 2 of the 3 ones, the last one is critical
        state.discard_pile.append(Card(color="red", number=1))
        state.discard_pile.append(Card(color="red", number=1))
        assert is_critical(state, Card(color="red", number=1))
        
        # Already played cards are not critical
        state.played_cards["blue"] = 3
        assert not is_critical(state, Card(color="blue", number=2))
