"""Tests for Hanabi visibility and information leak prevention."""

import pytest
from src.hanabi.models import (
    Card,
    CardKnowledge,
    HanabiConfig,
    HanabiState,
    HintAction,
    PlayAction,
)
from src.hanabi.game import create_game, apply_action
from src.hanabi.visibility import (
    view_for_player,
    assert_no_leaks,
    assert_view_safe,
    FORBIDDEN_KEYS,
)


def create_test_state() -> HanabiState:
    """Create a test game state."""
    config = HanabiConfig(num_players=3, seed=42)
    return create_game(config)


class TestViewForPlayer:
    """Tests for the view_for_player function."""

    def test_player_cannot_see_own_hand(self):
        """Critical: player must not see their own cards."""
        state = create_test_state()
        
        for player_id in state.player_order:
            view = view_for_player(state, player_id)
            
            # Player's own ID should NOT be in visible_hands
            assert player_id not in view["visible_hands"], \
                f"Player {player_id} can see their own hand - INFORMATION LEAK!"
            
            # But should have knowledge about their hand
            assert "my_hand_knowledge" in view
            assert len(view["my_hand_knowledge"]) == len(state.hands[player_id])

    def test_player_sees_all_other_hands(self):
        """Player should see all other players' hands."""
        state = create_test_state()
        
        for player_id in state.player_order:
            view = view_for_player(state, player_id)
            visible_players = set(view["visible_hands"].keys())
            expected_visible = set(state.player_order) - {player_id}
            
            assert visible_players == expected_visible, \
                f"Player {player_id} should see {expected_visible}, but sees {visible_players}"

    def test_other_hands_contain_actual_cards(self):
        """Visible hands should contain actual card data."""
        state = create_test_state()
        view = view_for_player(state, "player_1")
        
        for pid, visible_hand in view["visible_hands"].items():
            actual_hand = state.hands[pid]
            assert len(visible_hand) == len(actual_hand)
            
            for i, card_data in enumerate(visible_hand):
                assert "color" in card_data
                assert "number" in card_data
                assert card_data["color"] == actual_hand[i].color
                assert card_data["number"] == actual_hand[i].number

    def test_my_hand_knowledge_has_no_actual_card_data(self):
        """Own hand knowledge must not contain actual card values."""
        state = create_test_state()
        view = view_for_player(state, "player_1")
        
        for knowledge in view["my_hand_knowledge"]:
            # Should not have direct color/number (those are for actual cards)
            # Knowledge has known_color, known_number, possible_colors, possible_numbers
            assert "color" not in knowledge or knowledge.get("color") is None
            assert "number" not in knowledge or knowledge.get("number") is None

    def test_deck_not_visible(self):
        """Deck contents must not be visible."""
        state = create_test_state()
        view = view_for_player(state, "player_1")
        
        assert "deck" not in view
        assert "deck_order" not in view
        
        # Should only see deck size
        assert "deck_remaining" in view
        assert isinstance(view["deck_remaining"], int)


class TestAssertNoLeaks:
    """Tests for the leak detection function."""

    def test_detects_forbidden_keys(self):
        """Should detect forbidden keys in payload."""
        for key in FORBIDDEN_KEYS:
            payload = {key: "some_value"}
            with pytest.raises(AssertionError, match="Forbidden key"):
                assert_no_leaks(payload)

    def test_detects_nested_forbidden_keys(self):
        """Should detect forbidden keys in nested structures."""
        payload = {
            "safe_key": {
                "nested": {
                    "deck": [1, 2, 3]  # Forbidden
                }
            }
        }
        with pytest.raises(AssertionError, match="deck"):
            assert_no_leaks(payload)

    def test_detects_forbidden_keys_in_lists(self):
        """Should detect forbidden keys in list items."""
        payload = {
            "items": [
                {"safe": True},
                {"seed": 42},  # Forbidden
            ]
        }
        with pytest.raises(AssertionError, match="seed"):
            assert_no_leaks(payload)

    def test_passes_clean_payload(self):
        """Should pass for clean payloads."""
        payload = {
            "player_id": "player_1",
            "visible_hands": {"player_2": [{"color": "red", "number": 1}]},
            "score": 10,
        }
        # Should not raise
        assert_no_leaks(payload)


class TestAssertViewSafe:
    """Tests for view safety validation."""

    def test_rejects_own_hand_in_visible_hands(self):
        """Should reject if player's own hand is in visible_hands."""
        view = {
            "role": "player",
            "player_id": "player_1",
            "visible_hands": {
                "player_1": [{"color": "red", "number": 1}],  # LEAK!
                "player_2": [{"color": "blue", "number": 2}],
            },
            "my_hand_knowledge": [],
        }
        with pytest.raises(AssertionError, match="LEAK"):
            assert_view_safe(view)

    def test_accepts_valid_view(self):
        """Should accept a properly constructed view."""
        state = create_test_state()
        view = view_for_player(state, "player_1")
        
        # Should not raise
        assert_view_safe(view)

    def test_all_views_are_safe(self):
        """All player views should pass safety check."""
        state = create_test_state()
        
        for player_id in state.player_order:
            view = view_for_player(state, player_id)
            assert_view_safe(view)  # Should not raise


class TestHintKnowledgeUpdate:
    """Tests that hints properly update knowledge without leaking."""

    def test_hint_updates_knowledge(self):
        """Hints should update the target player's knowledge."""
        state = create_test_state()
        
        # Give a color hint from player_1 to player_2
        hint = HintAction(
            target_player="player_2",
            hint_type="color",
            hint_value="red",
        )
        
        # Find if player_2 has any red cards
        has_red = any(c.color == "red" for c in state.hands["player_2"])
        
        if has_red:
            new_state, result, _ = apply_action(state, "player_1", hint)
            
            # Check knowledge was updated
            assert result.success
            
            view = view_for_player(new_state, "player_2")
            
            # Player 2's knowledge should reflect the hint
            for i, k in enumerate(view["my_hand_knowledge"]):
                actual_card = new_state.hands["player_2"][i]
                if actual_card.color == "red":
                    assert k["known_color"] == "red"
                else:
                    assert "red" not in k["possible_colors"]

    def test_hint_in_history_is_visible_to_all(self):
        """Hints given should be visible in action history to all players."""
        state = create_test_state()
        
        # Find a valid hint
        target_hand = state.hands["player_2"]
        hint_color = target_hand[0].color  # First card's color
        
        hint = HintAction(
            target_player="player_2",
            hint_type="color",
            hint_value=hint_color,
        )
        
        new_state, result, _ = apply_action(state, "player_1", hint)
        
        if result.success:
            # All players should see this hint in history
            for player_id in new_state.player_order:
                view = view_for_player(new_state, player_id)
                assert len(view["action_history"]) == 1
                
                hist_action = view["action_history"][0]["action"]
                assert hist_action["action_type"] == "hint"
                assert hist_action["target_player"] == "player_2"
                assert hist_action["hint_type"] == "color"


class TestPlayResultVisibility:
    """Tests that play results are properly visible."""

    def test_play_result_reveals_card(self):
        """When a card is played, everyone should see what it was."""
        state = create_test_state()
        
        play = PlayAction(card_position=0)
        new_state, result, _ = apply_action(state, "player_1", play)
        
        if result.success:
            # All players should see the played card in history
            for player_id in new_state.player_order:
                view = view_for_player(new_state, player_id)
                hist = view["action_history"][0]
                
                # Result should contain the actual card played
                assert hist["result"]["card_played"] is not None
                assert "color" in hist["result"]["card_played"]
                assert "number" in hist["result"]["card_played"]


class TestStateConsistency:
    """Tests for state consistency across views."""

    def test_public_state_consistent_across_views(self):
        """Public game state should be identical for all players."""
        state = create_test_state()
        
        views = {pid: view_for_player(state, pid) for pid in state.player_order}
        
        # These should be identical across all views
        public_keys = [
            "played_cards",
            "discard_pile",
            "deck_remaining",
            "hint_tokens",
            "fuse_tokens",
            "score",
            "turn_number",
            "current_player",
            "player_order",
        ]
        
        first_view = views[state.player_order[0]]
        for player_id, view in views.items():
            for key in public_keys:
                assert view[key] == first_view[key], \
                    f"Public key {key} differs for {player_id}"
