"""Parsing utilities for Hanabi agent responses."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    Action,
    DiscardAction,
    HintAction,
    PlayAction,
    COLORS,
    NUMBERS,
)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract and parse a JSON object from text.
    Handles markdown code blocks and raw JSON.
    """
    # Try to find JSON in code blocks first (more lenient pattern)
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Find the first { and try to find matching }
    start = text.find('{')
    if start != -1:
        # Find balanced braces
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        
        if end > start:
            json_str = text[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try cleaning up common issues
                cleaned = json_str.replace('\n', ' ').replace('\r', '')
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
    
    # Try the whole text stripped
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Last resort: try to extract key-value pairs manually
    action_match = re.search(r'"action_type"\s*:\s*"(\w+)"', text)
    if action_match:
        action_type = action_match.group(1)
        result: dict[str, Any] = {"action_type": action_type}
        
        # Extract card_position
        pos_match = re.search(r'"card_position"\s*:\s*(\d+)', text)
        if pos_match:
            result["card_position"] = int(pos_match.group(1))
        
        # Extract target_player
        target_match = re.search(r'"target_player"\s*:\s*"([^"]+)"', text)
        if target_match:
            result["target_player"] = target_match.group(1)
        
        # Extract hint_type
        hint_type_match = re.search(r'"hint_type"\s*:\s*"(\w+)"', text)
        if hint_type_match:
            result["hint_type"] = hint_type_match.group(1)
        
        # Extract hint_value (string or number)
        hint_val_match = re.search(r'"hint_value"\s*:\s*(?:"([^"]+)"|(\d+))', text)
        if hint_val_match:
            result["hint_value"] = hint_val_match.group(1) or int(hint_val_match.group(2))
        
        # Extract rationale
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text)
        if rationale_match:
            result["rationale"] = rationale_match.group(1).replace('\\"', '"')
        
        return result
    
    return None


def parse_action_response(text: str) -> tuple[Action | None, str | None]:
    """
    Parse an action from an LLM response.
    
    Expected format:
    {
        "action_type": "play" | "discard" | "hint",
        "card_position": 0-4,  // for play/discard
        "target_player": "player_2",  // for hint
        "hint_type": "color" | "number",  // for hint
        "hint_value": "red" | 3,  // for hint
        "rationale": "..."
    }
    
    Returns:
        (action, error_message)
    """
    obj = parse_json_object(text)
    if obj is None:
        return None, "Failed to parse JSON from response"
    
    action_type = obj.get("action_type")
    
    if action_type == "play":
        pos = obj.get("card_position")
        if not isinstance(pos, int) or pos < 0:
            return None, f"Invalid card_position for play: {pos}"
        return PlayAction(card_position=pos), None
    
    elif action_type == "discard":
        pos = obj.get("card_position")
        if not isinstance(pos, int) or pos < 0:
            return None, f"Invalid card_position for discard: {pos}"
        return DiscardAction(card_position=pos), None
    
    elif action_type == "hint":
        target = obj.get("target_player")
        hint_type = obj.get("hint_type")
        hint_value = obj.get("hint_value")
        
        if not isinstance(target, str) or not target:
            return None, f"Invalid target_player: {target}"
        
        if hint_type not in ("color", "number"):
            return None, f"Invalid hint_type: {hint_type}"
        
        if hint_type == "color":
            if hint_value not in COLORS:
                return None, f"Invalid color hint_value: {hint_value}"
        else:
            try:
                hint_value = int(hint_value)
                if hint_value not in NUMBERS:
                    return None, f"Invalid number hint_value: {hint_value}"
            except (TypeError, ValueError):
                return None, f"Invalid number hint_value: {hint_value}"
        
        return HintAction(
            target_player=target,
            hint_type=hint_type,
            hint_value=hint_value,
        ), None
    
    else:
        return None, f"Unknown action_type: {action_type}"


def extract_rationale(text: str) -> str:
    """Extract rationale from response."""
    obj = parse_json_object(text)
    if obj and isinstance(obj.get("rationale"), str):
        return obj["rationale"]
    
    # Try to find RATIONALE: or REASONING: pattern
    match = re.search(r"(?:RATIONALE|REASONING)\s*:\s*(.+?)(?:\n\n|SCRATCHPAD|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def validate_hint(
    target_player: str,
    hint_type: str,
    hint_value: Any,
    hands: dict[str, list[dict[str, Any]]],
    player_id: str,
) -> tuple[bool, str | None]:
    """
    Validate a hint before applying.
    
    Returns:
        (is_valid, error_message)
    """
    # Can't hint yourself
    if target_player == player_id:
        return False, "Cannot give a hint to yourself"
    
    # Target must exist
    if target_player not in hands:
        return False, f"Unknown player: {target_player}"
    
    # Validate hint type and value
    if hint_type not in ("color", "number"):
        return False, f"Invalid hint type: {hint_type}"
    
    if hint_type == "color" and hint_value not in COLORS:
        return False, f"Invalid color: {hint_value}"
    
    if hint_type == "number":
        try:
            num = int(hint_value)
            if num not in NUMBERS:
                return False, f"Invalid number: {hint_value}"
        except (TypeError, ValueError):
            return False, f"Invalid number: {hint_value}"
    
    # Check hint touches at least one card
    target_hand = hands[target_player]
    if hint_type == "color":
        touches = any(card.get("color") == hint_value for card in target_hand)
    else:
        touches = any(card.get("number") == int(hint_value) for card in target_hand)
    
    if not touches:
        return False, f"Hint must touch at least one card"
    
    return True, None


def repair_prompt(error: str) -> str:
    """Generate a repair prompt for invalid responses."""
    return f"""Your previous response was invalid: {error}

Please provide a valid action in JSON format:

For PLAY:
{{"action_type": "play", "card_position": <0-4>, "rationale": "..."}}

For DISCARD:
{{"action_type": "discard", "card_position": <0-4>, "rationale": "..."}}

For HINT:
{{"action_type": "hint", "target_player": "<player_id>", "hint_type": "color"|"number", "hint_value": "<color>"|<number>, "rationale": "..."}}

Respond with ONLY the JSON object."""


def fallback_action(hint_tokens: int, hand_size: int) -> Action:
    """
    Generate a safe fallback action when parsing fails.
    
    Strategy: If hints available, discard oldest card. Otherwise play oldest.
    """
    if hint_tokens < 8:  # Can discard
        return DiscardAction(card_position=0)
    else:
        return PlayAction(card_position=0)
