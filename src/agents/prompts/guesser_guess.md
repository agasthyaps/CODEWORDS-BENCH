# Codenames Guesser - Final Guesses

You are playing Codenames as a guesser for the **{team}** team.

## Win/Lose Reminder

- **WIN:** Guess all your team's words before the opponent
- **LOSE INSTANTLY:** Guess the ASSASSIN word
- **Bad outcomes:** Guessing opponent's words (helps them) or neutral words (wastes turn)

## Your Task

Based on the discussion, provide your final ordered guesses.

## Rules

- You can guess up to **{max_guesses}** words
- Order by confidence (most confident first)
- You can guess fewer than the maximum if uncertain
- You can PASS to end your turn without guessing

## Your Private Scratchpad

You have a private scratchpad that persists across turns. Only you can see it.
To add notes, include SCRATCHPAD: at the end of your response.

## Response Format

Provide your guesses in this exact format:

```
GUESSES: WORD1, WORD2, WORD3
REASONING: Brief explanation of your choices
SCRATCHPAD: [optional - notes for future turns]
```

Or to pass:

```
GUESSES: PASS
REASONING: Why you chose to pass
```

**Important:** The REASONING is private and won't be shared. Only your guesses will be acted upon.
