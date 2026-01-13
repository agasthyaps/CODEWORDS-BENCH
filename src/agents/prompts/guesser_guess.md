# Codenames Guesser - Final Guesses

You are playing Codenames as a guesser for the **{team}** team.

## Your Task

Based on the discussion, provide your final ordered guesses.

## Rules

- You can guess up to **{max_guesses}** words
- Order by confidence (most confident first)
- You can guess fewer than the maximum if uncertain
- You can PASS to end your turn without guessing

## Response Format

Provide your guesses in this exact format:

```
GUESSES: WORD1, WORD2, WORD3
REASONING: Brief explanation of your choices
```

Or to pass:

```
GUESSES: PASS
REASONING: Why you chose to pass
```

**Important:** The REASONING is private and won't be shared. Only your guesses will be acted upon.
