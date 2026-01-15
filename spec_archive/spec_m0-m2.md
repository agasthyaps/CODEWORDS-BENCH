# **CODENAMES BENCHMARK: Implementation Spec**

### **Project Overview**

A benchmark for testing **adversarial multi-agent coordination**. Two 3-agent teams (1 clue-giver + 2 guessers each) play Codenames with full transcript visibility.

**Core constraint:** Everyone hears everything except the key. This is "strategic communication under surveillance." Deception may emerge—we observe, don't forbid.

**End state (M4+):** 6 agents, 2 teams, full adversarial play, metrics collection.

**This spec:** Foundation (M0-M2) that accommodates that complexity.

---

### **Architecture Principles**

1. **Public transcript vs private traces** — Strict separation. Public transcript = what agents see. Private traces = full LLM I/O, reasoning, retries. Never leak private into public.

2. **Seeding + episode IDs** — Every game reproducible. Board generation seeded, all randomness captured.

3. **Visibility via function, not convention** — `get_visible_state(role)` is the single source of truth for what each agent sees.

4. **Transcript is append-only** — Events have `turn_number` + `event_index` for ordering. Multiple events per turn.

5. **Invalid actions end turns, not retry loops** — For guesses. Cluers get retries (bounded). Benchmark stability over coddling.

---

### **Information Visibility Matrix**

| Information | Cluers (both) | Guessers (all 4) |
|-------------|---------------|------------------|
| Board (25 words) | ✓ | ✓ |
| Key (targets, assassin) | ✓ | ✗ |
| All clues (both teams) | ✓ | ✓ |
| All discussion (both teams) | ✓ | ✓ |
| All guesses + results | ✓ | ✓ |

**The ONLY asymmetry: guessers don't see the key.**

---

## **Milestone 0: Game Engine**

**Goal:** Deterministic game logic, no LLM calls, fully testable.

### **Data Model**

**Enums:**
- `Team`: RED, BLUE
- `CardType`: RED, BLUE, NEUTRAL, ASSASSIN
- `Phase`: CLUE, DISCUSSION, GUESS, GAME_OVER

**GameConfig:**
- `words_per_board`: 25
- `red_count`: 9 (first team advantage)
- `blue_count`: 8
- `neutral_count`: 7
- `assassin_count`: 1
- `starting_team`: RED
- `allow_unlimited_clue`: bool (support 0 and -1)
- `max_clue_number`: 9
- `seed`: optional int

**Board:**
- `words`: list[str] — 25 words, fixed order (position matters for future vision mode)
- `key_by_category`: dict mapping "red"/"blue"/"neutral"/"assassin" → set of words
- `key_by_word`: dict mapping word → CardType (O(1) lookup)

**Transcript Events (all have `turn_number` + `event_index`):**
- `Clue`: team, word, number
- `Guess`: team, word, result (what it actually was)
- `Pass`: team (explicit end-of-guessing)
- `DiscussionMessage`: team, agent_id, content

**GameState:**
- `config`, `board`, `board_seed`
- `revealed`: dict[word → CardType]
- `current_turn`, `phase`, `turn_number`, `event_counter`
- `current_clue`, `guesses_remaining`
- `public_transcript`: list of events
- `winner`: optional Team

**Private (separate from GameState):**
- `AgentTrace`: agent_id, turn, prompt_sent, raw_response, parsed_result, validation_errors, retry_count, model, temperature, latency, tokens
- `EpisodeRecord`: episode_id, config, board_seed, public_transcript, list[AgentTrace], winner, final_metrics

### **Core Functions**

**`generate_board(word_list, config) → (Board, seed_used)`**
- Uses config.seed if provided, else random
- Returns seed for reproducibility
- Distribution must match config (9/8/7/1)

**`validate_clue(word, number, state) → (bool, error_message)`**

Invalid if:
- Exact match to board word (case-insensitive)
- Clue is substring of board word OR board word is substring of clue
- Was used as previous clue (either team)
- Contains non-alpha characters
- Number out of range (respecting unlimited setting)

**`process_guess(word, state) → (new_state, result, turn_continues)`**

- Invalid guess (not on board, already revealed) = turn ends, nothing revealed
- Valid guess: reveal card, check winner, determine if turn continues
- Correct + guesses remaining → continue
- Wrong (opponent/neutral) → turn ends
- Assassin → game over, other team wins

**`process_pass(state) → new_state`**
- Adds Pass event to transcript
- Ends turn

**`check_winner(state) → optional Team`**
- All team words revealed → that team wins
- Assassin revealed → team that revealed it loses

**`get_visible_state(state, role) → dict`**
- Roles: "red_cluer", "blue_cluer", "red_guesser_1", "red_guesser_2", "blue_guesser_1", "blue_guesser_2"
- Cluers get `key` field
- Guessers get NO `key` field (not null—absent)
- Everyone gets full `public_transcript`

### **M0 Tests**

**Board generation:**
- 25 unique words
- Correct distribution (9/8/7/1)
- Seeding produces identical boards
- Both key representations consistent

**Clue validation:**
- Rejects exact board word
- Rejects case variations
- Rejects substring (both directions): "BANK" invalid if "BANKER" on board, "PINEAPPLE" invalid if "APPLE" on board
- Rejects previous clues
- Rejects special characters
- Accepts valid clue
- Handles 0 and unlimited correctly

**Guess processing:**
- Correct guess reveals + continues
- Decrements guesses_remaining
- Neutral ends turn
- Opponent word ends turn (+ can trigger their win)
- Assassin ends game
- Invalid guess (not on board) ends turn silently
- Already-revealed guess ends turn silently
- Exhausted guesses ends turn

**Pass:**
- Ends turn
- Recorded in transcript

**Win conditions:**
- All words revealed → win
- Opponent hits assassin → win
- No winner mid-game

**Visibility:**
- Cluer sees key
- Guesser has no key field
- Everyone sees full transcript

**Transcript:**
- Events ordered by event_index
- Multiple events same turn have distinct indices
- Serialization roundtrip works

**Full game (scripted):**
- Can play through with hardcoded moves
- Correct winner determined
- No moves accepted after game over

---

## **Milestone 1: Clue-Giver Agent**

**Goal:** One LLM agent produces legal clues.

### **AgentConfig**
- `model`: string
- `role`: "cluer" | "guesser"
- `team`: Team
- `agent_id`: string
- `temperature`: float
- `max_retries`: int (default 3)

### **CluerAgent**

**Method:** `generate_clue(visible_state) → (Clue, AgentTrace)`

**Flow:**
1. Format prompt with visible_state
2. Call LLM
3. Parse response (extract word, number, reasoning)
4. Sanitize (uppercase, strip punctuation)
5. Validate against game state
6. If invalid: add error to prompt, retry (up to max_retries)
7. If valid: return Clue + Trace
8. If exhausted: raise error

**Prompt must include:**
- Full rules (one word, number meaning, restrictions)
- Board words + revealed status
- Key (your words, opponent words, neutral, assassin)
- Remaining words for your team
- Full game transcript
- Strategic note: "Opponents hear your teammates' discussion and see your clue"

**Response format:**
```
CLUE: [word]
NUMBER: [integer or UNLIMITED]
REASONING: [private thinking]
```

**Parsing must handle:**
- Case variations
- Brackets around values
- Trailing punctuation
- UNLIMITED keyword → -1

**Reasoning goes to private trace only, never public transcript.**

### **M1 Tests**

**Parsing:**
- Standard format works
- Case insensitive
- Strips punctuation
- Handles UNLIMITED
- Handles 0
- Missing CLUE returns None
- Missing NUMBER returns None

**Agent:**
- Produces legal clue
- Retries on invalid (mock invalid first response)
- Raises after max retries
- Trace captures full interaction
- Clue word normalized to uppercase

**Integration (slow, actual LLM):**
- 10 boards, all legal clues
- Clues semantically related to targets (qualitative)

---

## **Milestone 2: Guesser Agent + Discussion**

**Goal:** Two guessers discuss and produce ordered guesses.

### **GuesserAgent**

**Method 1:** `discuss(visible_state, discussion_so_far) → (DiscussionMessage, AgentTrace)`
- Adds one message to discussion
- No retry (freeform natural language)
- Checks for consensus signal

**Method 2:** `make_guesses(visible_state, discussion) → (list[str], AgentTrace)`
- Produces final ordered guesses
- Validates + truncates (no retry)

### **Discussion Flow**

**`run_discussion(guessers, state, max_rounds=3) → list[DiscussionMessage]`**

- Guessers alternate (guesser_1, guesser_2, guesser_1, ...)
- Each message added to `public_transcript` immediately
- Ends when: 2 consecutive `CONSENSUS: YES` signals, OR max_rounds reached
- Returns full discussion

### **Consensus Mechanism**

Explicit tag in message:
```
CONSENSUS: YES
TOP: word1, word2, word3
```

Parser looks for `CONSENSUS: YES` (case-insensitive). Not vibes—parseable.

### **Guess Validation (No Retry)**

For each guess in order:
- Must be on board (case-insensitive match)
- Must not be already revealed
- Must not be duplicate in list

On first invalid: truncate list there (don't continue).

Max guesses = clue_number + 1 (or 25 if unlimited).

### **PASS Support**

`GUESSES: PASS` is valid. Returns empty list. Team ends turn voluntarily.

### **Discussion Prompt Must Include**

- Board + revealed
- Current clue (word + number)
- Full game transcript (both teams' history)
- Current discussion so far
- **Explicit warning: "The opposing team can read this entire discussion. The opposing clue-giver is listening."**
- Instructions: conversational, 1-4 sentences, signal consensus when ready

### **Final Guess Prompt Must Include**

- Same context
- Full discussion
- Max guesses allowed
- Instructions: order by confidence, can guess fewer, can PASS
- Format: `GUESSES: word1, word2 / REASONING: ...`

### **M2 Tests**

**Parsing:**
- Standard guess format
- Handles brackets
- Handles PASS
- Strips punctuation
- Truncates at non-board word (doesn't error)

**Discussion flow:**
- Two guessers alternate
- Messages added to public transcript
- Consensus signal ends early
- Max rounds caps discussion

**Guess validation:**
- Truncates at invalid word
- Truncates at revealed word
- Removes duplicates (skip, don't truncate)
- Respects max guesses
- PASS returns empty list

**Integration (slow, actual LLM):**
- Guesses relate to clue
- Discussion references clue word
- Discussion acknowledges surveillance (qualitative—do they ever hedge or misdirect?)

---

## **File Structure**

```
codenames-benchmark/
├── src/
│   ├── engine/          # M0: Board, state, rules, transcript
│   ├── agents/          # M1-2: CluerAgent, GuesserAgent
│   │   └── prompts/     # Prompt templates (text files)
│   ├── runner/          # M3+: Game loop, turn orchestration
│   └── metrics/         # M5+: Collection, analysis
├── tests/
│   ├── test_engine.py
│   ├── test_cluer.py
│   └── test_guesser.py
├── data/
│   └── wordlist.txt     # Official Codenames words
└── episodes/            # Saved game records (JSON)
```

---

## **Word List**

Use official Codenames words or equivalent curated list (~400 words). Words chosen for semantic ambiguity: BANK, SPRING, LEAD, MATCH, PALM, etc.

Source: BGG has community word lists. Don't use random English words—ambiguity is the game.

---

## **What's Deferred**

- **M3:** Turn orchestration, single-team full game
- **M4:** Two-team adversarial play
- **M5:** Metrics harness, benchmark runs
- **M6:** Live viewer
- **Future toggles:** Vision mode, blind mode (cluers infer targets), stemming validation, provider abstraction

---

## **Key Decisions Baked In**

| Decision | Rationale |
|----------|-----------|
| Public/private transcript split | Prevents accidental leakage |
| Seeding everything | Reproducibility for benchmark validity |
| Invalid guess = turn end (no retry) | Benchmark stability, measures actual capability |
| Explicit consensus tag | Parseable > vibes-based |
| Surveillance warning in prompt | Core to "adversarial coordination" framing |
| Deception allowed (not forbidden) | Let strategy emerge, measure it |
| Truncate invalid guesses | Graceful degradation, no infinite retry loops |
