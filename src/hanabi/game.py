"""Core game logic for Hanabi."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from .models import (
    Action,
    ActionResult,
    Card,
    CardKnowledge,
    Color,
    COLORS,
    CARD_COUNTS,
    DiscardAction,
    HanabiConfig,
    HanabiState,
    HintAction,
    Number,
    NUMBERS,
    PlayAction,
    TurnLog,
)


def create_deck(seed: int) -> list[Card]:
    """Create and shuffle a standard Hanabi deck."""
    rng = random.Random(seed)
    deck: list[Card] = []
    
    for color in COLORS:
        for number, count in CARD_COUNTS.items():
            for _ in range(count):
                deck.append(Card(color=color, number=number))  # type: ignore[arg-type]
    
    rng.shuffle(deck)
    return deck


def create_game(config: HanabiConfig, player_ids: list[str] | None = None) -> HanabiState:
    """
    Create a new Hanabi game.
    
    Args:
        config: Game configuration
        player_ids: Optional list of player IDs. If None, uses player_1, player_2, etc.
    
    Returns:
        Initial game state with dealt hands
    """
    seed = config.seed if config.seed is not None else random.randint(0, 2**31 - 1)
    
    # Store the resolved seed in config for later retrieval
    config = config.model_copy(update={"seed": seed})
    
    if player_ids is None:
        player_ids = [f"player_{i+1}" for i in range(config.num_players)]
    
    if len(player_ids) != config.num_players:
        raise ValueError(f"Expected {config.num_players} players, got {len(player_ids)}")
    
    deck = create_deck(seed)
    
    # Deal hands
    hands: dict[str, list[Card]] = {pid: [] for pid in player_ids}
    knowledge: dict[str, list[CardKnowledge]] = {pid: [] for pid in player_ids}
    
    for _ in range(config.hand_size):
        for pid in player_ids:
            if deck:
                hands[pid].append(deck.pop())
                knowledge[pid].append(CardKnowledge())
    
    # Initialize played cards (0 for each color)
    played_cards: dict[Color, int] = {color: 0 for color in COLORS}
    
    return HanabiState(
        config=config,
        hands=hands,
        knowledge=knowledge,
        played_cards=played_cards,
        discard_pile=[],
        deck=deck,
        hint_tokens=config.max_hints,
        fuse_tokens=config.max_fuses,
        current_player_idx=0,
        player_order=player_ids,
        turn_number=1,
        action_history=[],
        final_round_player=None,
        game_over=False,
        game_over_reason=None,
    )


def deal_card(state: HanabiState, player_id: str) -> HanabiState:
    """
    Deal a card to a player from the deck.
    
    If deck is empty, marks the start of the final round.
    """
    new_state = state.model_copy(deep=True)
    
    if new_state.deck:
        card = new_state.deck.pop()
        new_state.hands[player_id].append(card)
        new_state.knowledge[player_id].append(CardKnowledge())
        
        # If this was the last card, mark final round
        if not new_state.deck and new_state.final_round_player is None:
            new_state.final_round_player = player_id
    
    return new_state


def is_playable(card: Card, played_cards: dict[Color, int]) -> bool:
    """Check if a card can be legally played."""
    current = played_cards[card.color]
    return card.number == current + 1


def apply_play(state: HanabiState, player_id: str, action: PlayAction) -> tuple[HanabiState, ActionResult]:
    """Apply a play action."""
    new_state = state.model_copy(deep=True)
    
    hand = new_state.hands[player_id]
    if action.card_position < 0 or action.card_position >= len(hand):
        return state, ActionResult(
            success=False,
            message=f"Invalid card position: {action.card_position}",
        )
    
    # Remove card from hand
    card = hand.pop(action.card_position)
    new_state.knowledge[player_id].pop(action.card_position)
    
    # Check if playable
    playable = is_playable(card, new_state.played_cards)
    
    if playable:
        # Successfully play card
        new_state.played_cards[card.color] = card.number
        
        # Bonus hint token for completing a stack (playing a 5)
        if card.number == 5 and new_state.hint_tokens < new_state.config.max_hints:
            new_state.hint_tokens += 1
        
        result = ActionResult(
            success=True,
            message=f"Played {card} successfully",
            card_played=card,
            was_playable=True,
        )
    else:
        # Failed play - lose a fuse token
        new_state.fuse_tokens -= 1
        new_state.discard_pile.append(card)
        
        result = ActionResult(
            success=True,  # Action was valid, just unsuccessful
            message=f"Played {card} but it was not playable (needed {new_state.played_cards[card.color] + 1}). Lost a fuse token.",
            card_played=card,
            was_playable=False,
        )
    
    # Draw a new card
    new_state = deal_card(new_state, player_id)
    
    return new_state, result


def apply_discard(state: HanabiState, player_id: str, action: DiscardAction) -> tuple[HanabiState, ActionResult]:
    """Apply a discard action."""
    new_state = state.model_copy(deep=True)
    
    # Can't discard if at max hints
    if new_state.hint_tokens >= new_state.config.max_hints:
        return state, ActionResult(
            success=False,
            message="Cannot discard when at maximum hint tokens",
        )
    
    hand = new_state.hands[player_id]
    if action.card_position < 0 or action.card_position >= len(hand):
        return state, ActionResult(
            success=False,
            message=f"Invalid card position: {action.card_position}",
        )
    
    # Remove card from hand
    card = hand.pop(action.card_position)
    new_state.knowledge[player_id].pop(action.card_position)
    
    # Add to discard pile
    new_state.discard_pile.append(card)
    
    # Gain a hint token
    new_state.hint_tokens += 1
    
    # Draw a new card
    new_state = deal_card(new_state, player_id)
    
    return new_state, ActionResult(
        success=True,
        message=f"Discarded {card}, gained a hint token",
        card_discarded=card,
    )


def apply_hint(state: HanabiState, player_id: str, action: HintAction) -> tuple[HanabiState, ActionResult]:
    """Apply a hint action."""
    new_state = state.model_copy(deep=True)
    
    # Check hint tokens
    if new_state.hint_tokens <= 0:
        return state, ActionResult(
            success=False,
            message="No hint tokens available",
        )
    
    # Can't hint yourself
    if action.target_player == player_id:
        return state, ActionResult(
            success=False,
            message="Cannot give a hint to yourself",
        )
    
    # Check target player exists
    if action.target_player not in new_state.hands:
        return state, ActionResult(
            success=False,
            message=f"Unknown player: {action.target_player}",
        )
    
    target_hand = new_state.hands[action.target_player]
    target_knowledge = new_state.knowledge[action.target_player]
    
    # Find matching cards
    positions_touched: list[int] = []
    
    if action.hint_type == "color":
        hint_color = action.hint_value
        if hint_color not in COLORS:
            return state, ActionResult(
                success=False,
                message=f"Invalid color: {hint_color}",
            )
        
        for i, card in enumerate(target_hand):
            if card.color == hint_color:
                positions_touched.append(i)
                # Update knowledge
                target_knowledge[i].known_color = hint_color  # type: ignore[assignment]
                target_knowledge[i].possible_colors = {hint_color}  # type: ignore[assignment]
            else:
                # Negative information
                target_knowledge[i].possible_colors.discard(hint_color)  # type: ignore[arg-type]
    
    else:  # number hint
        hint_number = int(action.hint_value)
        if hint_number not in NUMBERS:
            return state, ActionResult(
                success=False,
                message=f"Invalid number: {hint_number}",
            )
        
        for i, card in enumerate(target_hand):
            if card.number == hint_number:
                positions_touched.append(i)
                # Update knowledge
                target_knowledge[i].known_number = hint_number  # type: ignore[assignment]
                target_knowledge[i].possible_numbers = {hint_number}  # type: ignore[assignment]
            else:
                # Negative information
                target_knowledge[i].possible_numbers.discard(hint_number)  # type: ignore[arg-type]
    
    # Hint must touch at least one card
    if not positions_touched:
        return state, ActionResult(
            success=False,
            message=f"Hint must touch at least one card. No cards match {action.hint_type}={action.hint_value}",
        )
    
    # Spend hint token
    new_state.hint_tokens -= 1
    
    return new_state, ActionResult(
        success=True,
        message=f"Hinted {action.target_player} about {action.hint_type}={action.hint_value}, touching positions {positions_touched}",
        positions_touched=positions_touched,
    )


def apply_action(
    state: HanabiState,
    player_id: str,
    action: Action,
    rationale: str = "",
) -> tuple[HanabiState, ActionResult, TurnLog]:
    """
    Apply an action to the game state.
    
    Returns:
        (new_state, result, turn_log)
    """
    if state.game_over:
        return state, ActionResult(success=False, message="Game is already over"), TurnLog(
            turn_number=state.turn_number,
            player_id=player_id,
            action=action,
            result=ActionResult(success=False, message="Game is already over"),
            rationale=rationale,
            hint_tokens_after=state.hint_tokens,
            fuse_tokens_after=state.fuse_tokens,
            score_after=state.score,
        )
    
    if player_id != state.current_player:
        return state, ActionResult(
            success=False,
            message=f"Not {player_id}'s turn (current: {state.current_player})",
        ), TurnLog(
            turn_number=state.turn_number,
            player_id=player_id,
            action=action,
            result=ActionResult(success=False, message=f"Not {player_id}'s turn"),
            rationale=rationale,
            hint_tokens_after=state.hint_tokens,
            fuse_tokens_after=state.fuse_tokens,
            score_after=state.score,
        )
    
    # Apply the specific action
    if isinstance(action, PlayAction):
        new_state, result = apply_play(state, player_id, action)
    elif isinstance(action, DiscardAction):
        new_state, result = apply_discard(state, player_id, action)
    elif isinstance(action, HintAction):
        new_state, result = apply_hint(state, player_id, action)
    else:
        return state, ActionResult(
            success=False,
            message=f"Unknown action type: {type(action)}",
        ), TurnLog(
            turn_number=state.turn_number,
            player_id=player_id,
            action=action,
            result=ActionResult(success=False, message="Unknown action type"),
            rationale=rationale,
            hint_tokens_after=state.hint_tokens,
            fuse_tokens_after=state.fuse_tokens,
            score_after=state.score,
        )
    
    if not result.success:
        return state, result, TurnLog(
            turn_number=state.turn_number,
            player_id=player_id,
            action=action,
            result=result,
            rationale=rationale,
            hint_tokens_after=state.hint_tokens,
            fuse_tokens_after=state.fuse_tokens,
            score_after=state.score,
        )
    
    # Create turn log
    turn_log = TurnLog(
        turn_number=new_state.turn_number,
        player_id=player_id,
        action=action,
        result=result,
        rationale=rationale,
        hint_tokens_after=new_state.hint_tokens,
        fuse_tokens_after=new_state.fuse_tokens,
        score_after=new_state.score,
    )
    
    # Add to history
    new_state.action_history.append(turn_log)
    
    # Check for game over conditions
    game_over, reason = check_terminal(new_state)
    if game_over:
        new_state.game_over = True
        new_state.game_over_reason = reason
    else:
        # Advance to next player
        new_state.current_player_idx = (new_state.current_player_idx + 1) % len(new_state.player_order)
        new_state.turn_number += 1
    
    return new_state, result, turn_log


def check_terminal(state: HanabiState) -> tuple[bool, str | None]:
    """
    Check if the game has ended.
    
    Returns:
        (is_game_over, reason)
        Reasons: "fuse_out", "perfect_score", "final_round_complete", None (not over)
    """
    # Fuse ran out
    if state.fuse_tokens <= 0:
        return True, "fuse_out"
    
    # Perfect score
    if state.score == 25:
        return True, "perfect_score"
    
    # Final round complete (everyone played once after deck emptied)
    if state.final_round_player is not None:
        # Check if we've gone around once since final_round_player
        final_idx = state.player_order.index(state.final_round_player)
        current_idx = state.current_player_idx
        
        # Count turns since the final card was drawn
        # The final round player plays, then everyone else plays, then game ends
        # when it would be the final_round_player's turn again
        if current_idx == final_idx and len(state.action_history) > 0:
            # Check that the last action wasn't by the final_round_player
            # (to avoid ending immediately when deck empties)
            if state.action_history[-1].player_id != state.final_round_player:
                return True, "final_round_complete"
    
    return False, None


def get_playable_cards(state: HanabiState) -> dict[Color, int]:
    """Get the next playable number for each color."""
    return {color: played + 1 for color, played in state.played_cards.items() if played < 5}


def count_remaining(state: HanabiState, card: Card) -> int:
    """Count how many copies of a card are still available (deck + hands)."""
    total = CARD_COUNTS[card.number]
    
    # Subtract discarded
    discarded = sum(1 for c in state.discard_pile if c.color == card.color and c.number == card.number)
    
    return total - discarded


def is_critical(state: HanabiState, card: Card) -> bool:
    """Check if a card is critical (last copy and still needed)."""
    # Already played or not needed
    if state.played_cards[card.color] >= card.number:
        return False
    
    # Is this the last copy?
    return count_remaining(state, card) == 1
