"""LLM-based Hanabi player agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.llm import LLMProvider
from src.core.parsing import extract_scratchpad, remove_scratchpad_from_response

from ..models import Action, HanabiState
from ..parsing import (
    extract_rationale,
    fallback_action,
    parse_action_response,
    repair_prompt,
)
from ..visibility import assert_view_safe, view_for_player


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = Path(__file__).parent / "prompts" / name
    with open(path, "r") as f:
        return f.read()


def _format_visible_hands(hands: dict[str, list[dict[str, Any]]]) -> str:
    """Format visible hands for display in prompt."""
    lines = []
    for player_id, cards in hands.items():
        card_strs = [f"{c['color'][0].upper()}{c['number']}" for c in cards]
        lines.append(f"  {player_id}: {' | '.join(card_strs)}")
    return "\n".join(lines)


def _format_my_knowledge(knowledge: list[dict[str, Any]]) -> str:
    """Format own hand knowledge for display in prompt."""
    lines = []
    for i, k in enumerate(knowledge):
        known_parts = []
        if k.get("known_color"):
            known_parts.append(f"color={k['known_color']}")
        if k.get("known_number"):
            known_parts.append(f"number={k['known_number']}")
        
        if known_parts:
            known_str = ", ".join(known_parts)
        else:
            possible_c = k.get("possible_colors", [])
            possible_n = k.get("possible_numbers", [])
            known_str = f"could be: colors={possible_c}, numbers={possible_n}"
        
        lines.append(f"  Position {i}: {known_str}")
    return "\n".join(lines)


def _format_played_cards(played: dict[str, int]) -> str:
    """Format played card stacks."""
    parts = []
    for color, num in played.items():
        if num > 0:
            parts.append(f"{color[0].upper()}:{num}")
        else:
            parts.append(f"{color[0].upper()}:—")
    return " | ".join(parts)


def _format_recent_actions(history: list[dict[str, Any]], limit: int = 30) -> str:
    """Format recent action history (default: ~10 rounds with 3 players)."""
    if not history:
        return "  (no actions yet)"

    recent = history[-limit:]
    lines = []
    for h in recent:
        action = h.get("action", {})
        result = h.get("result", {})
        player = h.get("player_id", "?")
        turn = h.get("turn_number", "?")
        
        action_type = action.get("action_type")
        if action_type == "play":
            card = result.get("card_played", {})
            success = "✓" if result.get("was_playable") else "✗"
            lines.append(f"  T{turn} {player}: PLAY pos {action.get('card_position')} → {card.get('color', '?')[0].upper()}{card.get('number', '?')} {success}")
        elif action_type == "discard":
            card = result.get("card_discarded", {})
            lines.append(f"  T{turn} {player}: DISCARD pos {action.get('card_position')} → {card.get('color', '?')[0].upper()}{card.get('number', '?')}")
        elif action_type == "hint":
            target = action.get("target_player")
            hint_type = action.get("hint_type")
            hint_value = action.get("hint_value")
            positions = result.get("positions_touched", [])
            lines.append(f"  T{turn} {player}: HINT {target} {hint_type}={hint_value} (positions {positions})")
    
    return "\n".join(lines)


@dataclass(frozen=True)
class HanabiPlayerLLM:
    """LLM-based Hanabi player."""
    
    provider: LLMProvider
    player_id: str
    temperature: float = 0.7

    async def decide_action(
        self,
        state: HanabiState,
        scratchpad_content: str = "",
    ) -> tuple[Action, str, str | None]:
        """
        Decide what action to take this turn.
        
        Args:
            state: Current game state
            scratchpad_content: Agent's private scratchpad from previous turns
            
        Returns:
            (action, rationale, scratchpad_addition)
        """
        # Build view for this player
        view = view_for_player(state, self.player_id)
        assert_view_safe(view)
        
        # Build prompt
        system = _load_prompt("player_system.md")
        
        # Format view components for readability
        view_summary = f"""## Current Game State (Turn {view['turn_number']})

### Other Players' Hands (YOU CAN SEE THESE)
{_format_visible_hands(view['visible_hands'])}

### Your Hand (YOU CANNOT SEE THESE - only hints received)
{_format_my_knowledge(view['my_hand_knowledge'])}
Hand size: {view['my_hand_size']} cards

### Played Cards (stacks on table)
{_format_played_cards(view['played_cards'])}
Current score: {view['score']}/25

### Next Playable Cards
{json.dumps(view['playable_next'])}

### Resources
Hint tokens: {view['hint_tokens']}/8
Fuse tokens: {view['fuse_tokens']}/3
Cards remaining in deck: {view['deck_remaining']}

### Discard Pile ({len(view['discard_pile'])} cards)
{', '.join(f"{c['color'][0].upper()}{c['number']}" for c in view['discard_pile']) or '(empty)'}

### Recent Actions
{_format_recent_actions(view['action_history'])}

### Turn Order
Players: {' → '.join(view['player_order'])}
Current player: {view['current_player']} {'(YOU)' if view['is_my_turn'] else ''}
"""
        
        scratchpad_section = ""
        if scratchpad_content:
            scratchpad_section = f"\n## Your Private Scratchpad\n{scratchpad_content}\n"
        
        user = _load_prompt("player_turn.md").format(
            view=view_summary,
            player_id=self.player_id,
            scratchpad=scratchpad_section,
            hint_tokens=view['hint_tokens'],
            other_players=", ".join(p for p in view['player_order'] if p != self.player_id),
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        # Get LLM response
        resp1 = await self.provider.complete(messages=messages, temperature=self.temperature)
        
        # Extract scratchpad addition
        scratchpad_addition = extract_scratchpad(resp1.content)
        
        # Parse action
        action, error = parse_action_response(resp1.content)
        rationale = extract_rationale(resp1.content)
        
        if action is not None:
            return action, rationale, scratchpad_addition
        
        # Retry once with repair prompt
        messages.append({"role": "assistant", "content": resp1.content})
        messages.append({"role": "user", "content": repair_prompt(error or "unknown error")})
        
        resp2 = await self.provider.complete(messages=messages, temperature=self.temperature)
        scratchpad_addition = extract_scratchpad(resp2.content) or scratchpad_addition
        
        action2, error2 = parse_action_response(resp2.content)
        rationale2 = extract_rationale(resp2.content)
        
        if action2 is not None:
            return action2, rationale2, scratchpad_addition
        
        # Fallback action
        fallback = fallback_action(view['hint_tokens'], view['my_hand_size'])
        return fallback, f"Fallback action due to parse failure: {error2}", scratchpad_addition
