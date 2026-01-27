# Clue Deliberation Step

You are about to give a clue to your teammates. Before committing, carefully consider multiple options.

## Current Game State

### Board Words
{board_words_display}

### The Key (Only You Can See This)

**Your team's words ({team}):** {your_words}
**Opponent's words ({opponent_team}):** {opponent_words}
**Neutral words:** {neutral_words}
**ASSASSIN (avoid at all costs):** {assassin_word}

### Remaining Words to Get

Your team still needs: {remaining_words}

### Game History

{transcript_display}

## Your Task

Think through multiple potential clues before choosing the best one.

### Step 1: Brainstorm Candidates

For each candidate clue, consider:
- Which of YOUR remaining words does it connect to?
- What words will your teammates likely guess (in order)?
- What confusion risks exist (opponent words, neutral, assassin)?

### Step 2: Choose Your Best Clue

Select the clue that maximizes correct guesses while minimizing risks.

## Response Format

```
CANDIDATES:
1. OCEAN (3): Targets [WAVE, FISH, BEACH]. Teammates will guess: [WAVE, FISH, BEACH]. Risks: [SHIP: opponent word]
2. WATER (2): Targets [WAVE, FISH]. Teammates will guess: [WAVE, FISH]. Risks: [POOL: neutral]
3. ANIMAL (2): Targets [FISH, CAT]. Teammates will guess: [FISH, CAT]. Risks: [DOG: assassin word]

CHOSEN_CLUE: [your one-word clue in UPPERCASE]
CHOSEN_NUMBER: [integer 0-9 or UNLIMITED]
PREDICTED_GUESSES: [WORD1, WORD2, ...] (board words your teammates will guess, in order)
TRANSLATED_GUESSES: [WORD1, WORD2, ...] (same as above - must be actual board words)
CONFUSION_RISKS:
- [WORDX]: [short reason why teammates might wrongly guess this]
- [WORDY]: [short reason]
CONFIDENCE: [1-5] (how confident are you teammates will guess correctly?)
REASONING: [why this clue is best among your candidates]
```

Remember:
- Your clue must be a single word with only letters
- It cannot match or contain any board word (or be contained by one)
- The opposing team hears everything - be mindful of information leakage
