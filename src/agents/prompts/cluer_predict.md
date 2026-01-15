# Prediction Step (Private)

You gave the clue: {clue_word}, {clue_number}

Before seeing your teammates' discussion, predict what they will do.

## Board Words (unrevealed)
{board_words_display}

Respond in exactly this format:

```
PREDICTED_GUESSES: WORD1, WORD2, WORD3
TRANSLATED_GUESSES: BOARDWORD1, BOARDWORD2, BOARDWORD3
CONFIDENCE: 1-5
CONFUSION_RISKS:
- WORDX: short reason
- WORDY: short reason
```

Notes:
- **PREDICTED_GUESSES** can be *concepts* (they may be synonyms, not necessarily on the board).
- **TRANSLATED_GUESSES** must be **board words only**, mapping your predicted concepts to the nearest board words.
- If you think they'll pass immediately, set `PREDICTED_GUESSES: PASS` and `TRANSLATED_GUESSES: PASS`.
