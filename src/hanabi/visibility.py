"""Visibility and view generation for Hanabi.

Core principle: A player can see ALL other players' hands but NOT their own cards.
They only know about their own cards through hints received.
"""

from __future__ import annotations

from typing import Any

from .models import HanabiState, TurnLog, Color


# Keys that must NEVER appear in any player view
FORBIDDEN_KEYS = {
    "deck",
    "deck_order",
    "rng",
    "seed",
    "random",
    "debug",
    "_internal",
}


def _public_turn_summary(turn: TurnLog) -> dict[str, Any]:
    """
    Create a public summary of a turn.
    All turn information is public in Hanabi.
    """
    action_dict = turn.action.model_dump()
    result_dict = turn.result.model_dump()
    
    return {
        "turn_number": turn.turn_number,
        "player_id": turn.player_id,
        "action": action_dict,
        "result": result_dict,
        "hint_tokens_after": turn.hint_tokens_after,
        "fuse_tokens_after": turn.fuse_tokens_after,
        "score_after": turn.score_after,
    }


def view_for_player(state: HanabiState, player_id: str) -> dict[str, Any]:
    """
    Build the dynamically redacted game state view for a specific player.
    
    CRITICAL: The player can see ALL other players' hands but NOT their own cards.
    They only see their CardKnowledge (what they've learned from hints).
    
    Args:
        state: Current game state
        player_id: The player requesting the view
        
    Returns:
        Redacted view dictionary safe for the player to see
    """
    if player_id not in state.hands:
        raise ValueError(f"Unknown player: {player_id}")
    
    # Build visible hands - all players EXCEPT the requesting player
    visible_hands: dict[str, list[dict[str, Any]]] = {}
    for pid, hand in state.hands.items():
        if pid != player_id:
            visible_hands[pid] = [
                {"color": card.color, "number": card.number}
                for card in hand
            ]
    
    # Player's own hand knowledge (from hints, not actual cards)
    my_hand_knowledge = [k.model_dump() for k in state.knowledge[player_id]]
    
    # Played cards (public)
    played_cards: dict[str, int] = {color: num for color, num in state.played_cards.items()}
    
    # Discard pile (public)
    discard_pile = [
        {"color": card.color, "number": card.number}
        for card in state.discard_pile
    ]
    
    # What cards are still playable
    playable_next: dict[str, int] = {
        color: played + 1
        for color, played in state.played_cards.items()
        if played < 5
    }
    
    # Full action history (all public)
    action_history = [_public_turn_summary(t) for t in state.action_history]
    
    return {
        "role": "player",
        "player_id": player_id,
        "turn_number": state.turn_number,
        
        # Other players' hands - VISIBLE
        "visible_hands": visible_hands,
        
        # Own hand - only knowledge from hints, NOT actual cards
        "my_hand_knowledge": my_hand_knowledge,
        "my_hand_size": len(state.hands[player_id]),
        
        # Public game state
        "played_cards": played_cards,
        "playable_next": playable_next,
        "discard_pile": discard_pile,
        "deck_remaining": len(state.deck),
        "hint_tokens": state.hint_tokens,
        "fuse_tokens": state.fuse_tokens,
        "score": state.score,
        
        # Action history (all public)
        "action_history": action_history,
        
        # Turn information
        "current_player": state.current_player,
        "player_order": list(state.player_order),
        "is_my_turn": state.current_player == player_id,
        
        # End game info
        "final_round_started": state.final_round_player is not None,
        "game_over": state.game_over,
    }


def assert_no_leaks(payload: Any, path: str = "") -> None:
    """
    Recursively assert that no forbidden keys appear in a payload.
    
    Raises AssertionError if any leak is detected.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key).lower()
            current_path = f"{path}.{key}" if path else key
            
            if key_str in FORBIDDEN_KEYS:
                raise AssertionError(f"Forbidden key '{key}' found at {current_path}")
            
            # Check for actual card data in player's own hand context
            # This shouldn't happen if view_for_player is correct
            if key_str == "my_hand" or key_str == "own_hand":
                raise AssertionError(f"Direct hand access found at {current_path}")
            
            assert_no_leaks(value, current_path)
    
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_no_leaks(item, f"{path}[{i}]")


def assert_view_safe(view: dict[str, Any]) -> None:
    """
    Validate that a player view is safe (no information leaks).
    
    Checks:
    1. No forbidden keys anywhere in the payload
    2. Player's own cards are not directly visible
    3. Only CardKnowledge is present for own hand
    """
    if not isinstance(view, dict):
        raise AssertionError("View must be a dictionary")
    
    if view.get("role") != "player":
        raise AssertionError(f"Unknown role in view: {view.get('role')}")
    
    player_id = view.get("player_id")
    if not player_id:
        raise AssertionError("View missing player_id")
    
    # Check player's own cards aren't in visible_hands
    visible_hands = view.get("visible_hands", {})
    if player_id in visible_hands:
        raise AssertionError(f"Player {player_id}'s own hand found in visible_hands - LEAK!")
    
    # Ensure my_hand_knowledge doesn't contain actual card data
    my_knowledge = view.get("my_hand_knowledge", [])
    for i, k in enumerate(my_knowledge):
        if isinstance(k, dict):
            # Knowledge should only have known_color, known_number, possible_colors, possible_numbers
            allowed_keys = {"known_color", "known_number", "possible_colors", "possible_numbers"}
            extra_keys = set(k.keys()) - allowed_keys
            if extra_keys:
                # Check if any leaked card info
                if "color" in extra_keys or "number" in extra_keys:
                    raise AssertionError(f"Actual card data found in my_hand_knowledge[{i}] - LEAK!")
    
    # Run general leak check
    assert_no_leaks(view)


def get_hint_targets(state: HanabiState, player_id: str) -> list[str]:
    """Get valid hint target players (everyone except self)."""
    return [pid for pid in state.player_order if pid != player_id]


def describe_hand_for_others(state: HanabiState, player_id: str) -> list[str]:
    """
    Describe a player's hand as others see it.
    Used for UI display.
    """
    return [str(card) for card in state.hands[player_id]]
