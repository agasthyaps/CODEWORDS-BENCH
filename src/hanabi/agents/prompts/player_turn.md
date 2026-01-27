{view}
{scratchpad}
---

You are **{player_id}**. It is your turn.

## Available Actions

1. **PLAY**: Play a card by position (0 = oldest, {hint_tokens} hint tokens available)
   - Risk: If not playable, lose a fuse token
   - Only play if you're confident the card is needed

2. **DISCARD**: Discard a card by position (regain 1 hint token)
   - Safe if you have duplicate or unneeded cards
   - Cannot discard if at max hints (8)

3. **HINT**: Give a hint to {other_players} (costs 1 hint token)
   - Use hints strategically to signal plays or saves
   - Hint tokens: {hint_tokens}/8

## Decision Framework

Consider:
1. **What do I know about my hand?** (from hints received)
2. **What cards are needed next?** (check playable_next)
3. **Should I play?** (only if confident a card is playable)
4. **Should I hint?** (to help a teammate play or save a critical card)
5. **Should I discard?** (if nothing urgent and need hint tokens)

## Your Response

Provide your action as a JSON object, then optionally add scratchpad notes:

```json
{{
  "action_type": "play" | "discard" | "hint",
  "card_position": <0-4>,  // for play/discard
  "target_player": "<player_id>",  // for hint
  "hint_type": "color" | "number",  // for hint
  "hint_value": "<value>",  // for hint
  "rationale": "Brief explanation of your reasoning"
}}
```

SCRATCHPAD: [optional notes for future turns]
