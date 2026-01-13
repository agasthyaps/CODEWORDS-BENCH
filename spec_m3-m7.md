## **Milestone 3: Turn Orchestration + Single-Team Play**

**Goal:** Complete game loop with one real team. Other team passes every turn (solo win condition: clear all 9 words).

### **TurnOrchestrator**

**`run_clue_phase(team, state) → (new_state, AgentTrace)`**
1. Get visible state for `{team}_cluer`
2. Call `cluer.generate_clue(visible_state)`
3. Add Clue event to `public_transcript`
4. Update `current_clue`, `guesses_remaining`, `phase`
5. Return new state + trace

**`run_discussion_phase(team, state, max_rounds=3) → (new_state, list[AgentTrace])`**
1. Get both guessers for team
2. Alternate: guesser_1, guesser_2, guesser_1...
3. Each call: `guesser.discuss(visible_state, discussion_so_far)`
4. Add DiscussionMessage to `public_transcript` immediately
5. Check for consecutive `CONSENSUS: YES` from both guessers
6. End on consensus OR max_rounds
7. Return new state + all traces

**`run_guess_phase(team, state, discussion) → (new_state, AgentTrace)`**
1. Designated guesser (guesser_1) calls `make_guesses(visible_state, discussion)`
2. For each guess in order:
   - Call `process_guess(guess, state)`
   - Add Guess event to transcript
   - If `turn_continues == False`: break
   - If `winner is not None`: break
3. If no guesses or PASS: add Pass event
4. Switch turn to other team
5. Return new state + trace

**`run_turn(team, state) → (new_state, TurnTraces)`**
```python
state, clue_trace = run_clue_phase(team, state)
state, discussion_traces = run_discussion_phase(team, state)
state, guess_trace = run_guess_phase(team, state)
return state, TurnTraces(clue_trace, discussion_traces, guess_trace)
```

### **GhostTeam**

For single-team testing, opponent team uses `GhostTeam`:

```python
GhostMode:
    PASS      # Always passes (no clue, no guesses) — solo mode
    RANDOM    # Random legal clue, random valid guesses — tests turn switching
```

**`GhostCluer.generate_clue(state) → Clue`**
- PASS mode: Returns `Clue("PASS", 0)` — special case, turn ends immediately
- RANDOM mode: Random word not on board, random number 1-3

**`GhostGuesser.discuss(...)` / `make_guesses(...)`**
- PASS mode: Returns empty discussion, empty guesses
- RANDOM mode: Random unrevealed words from board

Ghost traces are still recorded (for consistency) but marked `is_ghost: true`.

### **EpisodeRunner**

**`run_episode(config, red_team, blue_team) → EpisodeRecord`**
```python
state = initialize_game(config)
all_traces = []

while state.winner is None:
    team = state.current_turn
    agents = red_team if team == RED else blue_team
    state, turn_traces = run_turn(team, state, agents)
    all_traces.append(turn_traces)

return EpisodeRecord(
    episode_id=uuid(),
    config=config,
    board_seed=state.board_seed,
    public_transcript=state.public_transcript,
    traces=all_traces,
    winner=state.winner
)
```

### **EpisodeRecord Schema**

```python
EpisodeRecord:
    episode_id: str
    config: GameConfig
    board_seed: int
    public_transcript: list[Event]
    traces: list[TurnTraces]  # Private, never in transcript
    winner: Team
    metadata: dict  # timestamps, model versions, etc.
```

**TurnTraces:**
```python
TurnTraces:
    turn_number: int
    team: Team
    clue_trace: AgentTrace
    discussion_traces: list[AgentTrace]
    guess_trace: AgentTrace
```

### **M3 Tests**

**Turn orchestration:**
- Clue phase updates state correctly
- Discussion messages appear in transcript in order
- Guess phase processes until turn ends
- Turn switches to other team after guess phase
- Phase transitions: CLUE → DISCUSSION → GUESS → CLUE (next team)

**Ghost team:**
- PASS ghost produces no events (or Pass event only)
- RANDOM ghost produces legal clues and valid guesses
- Ghost traces marked correctly

**Single-team game (vs PASS ghost):**
- Real team plays, ghost passes every turn
- Game ends when real team clears all 9 words
- Game ends if real team hits assassin
- Win recorded correctly

**Episode record:**
- Serializes to JSON
- Deserializes back to identical object
- Transcript and traces are separate
- Can reconstruct game state from transcript alone

**Replay determinism:**
- Same seed + same agent responses = identical transcript
- Verify by mocking LLM responses

---

## **Milestone 4: Two-Team Adversarial Play**

**Goal:** Full 6-agent game. Both teams have real agents. This is the core benchmark.

### **TeamConfig**

```python
TeamConfig:
    cluer: AgentConfig
    guesser_1: AgentConfig
    guesser_2: AgentConfig
```

### **Turn Alternation**

Standard Codenames rules:
1. RED clues → RED discusses → RED guesses → check winner
2. BLUE clues → BLUE discusses → BLUE guesses → check winner
3. Repeat until winner

Starting team (RED) has 9 words, other team (BLUE) has 8. First-move advantage balanced by target count.

### **Cross-Team Visibility (Exercised)**

Already defined in M0's visibility matrix, now actively used:

- When RED cluer generates clue, `visible_state` includes BLUE's previous discussion
- When BLUE guessers discuss, they know RED heard their last discussion
- Surveillance warning in prompts is now strategically relevant

**Observable dynamics (measure, don't enforce):**
- Clue adaptation: Does cluer's strategy shift after reading opponent discussion?
- Discussion hedging: Do guessers avoid explicit word mentions?
- Misdirection: Do guessers discuss words they don't intend to guess?

### **Game Modes**

```python
GameMode:
    STANDARD        # Full rules, assassin active
    NO_ASSASSIN     # Assassin card treated as neutral (cleaner signal)
    SINGLE_GUESSER  # 1 guesser per team (ablation: consensus value)
```

**NO_ASSASSIN implementation:**
- `assassin_count: 0` in config
- Extra neutral card instead
- Removes instant-loss variance

**SINGLE_GUESSER implementation:**
- Skip discussion phase entirely
- Single guesser goes straight to `make_guesses()`
- Allows measuring: does discussion help or hurt?

### **M4 Tests**

**Two-team game:**
- Both teams take turns
- Game completes with winner
- Turn count is reasonable (not infinite loops)

**Visibility enforcement:**
- Guesser's `visible_state` never contains `key` field (not null—absent)
- Cluer's `visible_state` contains opponent's discussion
- Verify with mock agents that log their inputs

**Game modes:**
- STANDARD: Assassin hit ends game, other team wins
- NO_ASSASSIN: Hitting that card just ends turn
- SINGLE_GUESSER: No discussion traces, single guesser decides alone

**Cross-team dynamics (qualitative, logged for analysis):**
- Log whether clue changed after opponent discussion (compare to baseline)
- Flag discussions that mention opponent's clue

**Edge cases:**
- Both teams down to 1 word each
- Assassin hit on first guess
- Team clears all words on bonus guess (+1)

---

## **Milestone 5: Metrics Harness**

**Goal:** Compute meaningful metrics from EpisodeRecords. Answer: "What makes coordination good?"

### **Per-Episode Metrics**

```python
EpisodeMetrics:
    episode_id: str
    winner: Team
    turns_to_win: int
    
    # Per team
    red_metrics: TeamMetrics
    blue_metrics: TeamMetrics
```

```python
TeamMetrics:
    words_cleared: int
    assassin_hit: bool
    
    # Clue metrics
    total_clues: int
    avg_clue_number: float          # Ambition
    clue_efficiency: float          # correct_guesses / sum(clue_numbers)
    
    # Guess metrics  
    total_guesses: int
    correct_guesses: int
    wrong_guesses: int              # Opponent or neutral
    guess_accuracy: float           # correct / total
    
    # Discussion metrics
    avg_discussion_rounds: float
    consensus_rate: float           # % of turns with explicit consensus
    avg_discussion_length: int      # Total tokens/chars
```

### **Derived Metrics**

**Coordination Score** (composite):
```python
coordination_score = (
    0.4 * clue_efficiency +
    0.3 * guess_accuracy +
    0.2 * consensus_rate +
    0.1 * (1 / avg_discussion_rounds)  # Faster consensus = better
)
```

Weights adjustable. This is a starting point.

**Theory of Mind Score:**

Requires parsing cluer's REASONING to extract intended targets.

```python
intended_targets = parse_intended_targets(clue_trace.reasoning)
actual_guesses = [g.word for g in guesses if g.result == CORRECT]
tom_score = len(set(intended_targets) & set(actual_guesses)) / len(intended_targets)
```

If REASONING doesn't reliably contain targets, fall back to: "Did guessers guess cluer's team's words?" (weaker signal).

**Surveillance Adaptation Score:**

Compare clue given BEFORE seeing opponent discussion (turn 1) vs AFTER (turn 2+).

Qualitative for now: flag episodes where cluer's reasoning mentions opponent discussion.

### **Aggregate Metrics (Across N Episodes)**

```python
AggregateMetrics:
    config: GameConfig
    episodes: int
    
    win_rate_red: float
    win_rate_blue: float
    
    avg_turns_to_win: float
    std_turns_to_win: float
    
    avg_coordination_score: float
    avg_theory_of_mind: float
    
    assassin_rate: float  # % of games ending in assassin hit
```

### **MetricsCollector**

**`compute_episode_metrics(episode: EpisodeRecord) → EpisodeMetrics`**
- Pure function, no side effects
- Derives all metrics from transcript + traces

**`compute_aggregate_metrics(episodes: list[EpisodeRecord]) → AggregateMetrics`**
- Aggregates across episodes
- Computes means, stds, rates

**`export_metrics(metrics, format="json") → str`**
- JSON for programmatic use
- CSV for spreadsheet analysis
- Markdown table for reports

### **M5 Tests**

**Episode metrics:**
- Correct winner extracted
- Turn count matches transcript
- Clue efficiency calculated correctly (manual example)
- Guess accuracy calculated correctly

**Edge cases:**
- Episode with 0 wrong guesses (perfect game)
- Episode ending on turn 1 (assassin hit)
- Episode with no consensus achieved (all timeouts)

**Aggregate metrics:**
- Win rates sum to 1.0 (no ties in Codenames)
- Averages computed correctly
- Handles single-episode input

**Theory of mind:**
- Parser extracts targets from well-formed REASONING
- Graceful fallback when REASONING is malformed
- Score in [0, 1] range

**Export:**
- JSON roundtrips correctly
- CSV has correct headers
- Markdown renders as table

---

## **Milestone 6: Benchmark Runs + Model Comparison**

**Goal:** Systematic experiments. Publishable results. Answer: "Which models coordinate best?"

### **ExperimentConfig**

```python
ExperimentConfig:
    name: str
    description: str
    
    # What to vary
    models: list[str]               # ["claude-sonnet", "gpt-4o", "gemini-pro"]
    game_modes: list[GameMode]      # [STANDARD, SINGLE_GUESSER]
    team_compositions: list[str]    # ["homogeneous", "mixed_cluer", "mixed_guesser"]
    
    # Fixed
    seeds: list[int]                # 100 pre-generated seeds
    games_per_config: int           # Games per (model, mode, composition, seed)
    
    # Optional
    temperature: float              # Default 0.7
    max_retries: int                # Default 3
```

### **Team Compositions**

```python
"homogeneous"     # All 6 agents same model
"mixed_cluer"     # Cluers are model A, guessers are model B
"mixed_guesser"   # Cluer + guesser_1 are model A, guesser_2 is model B
"heterogeneous"   # All different (if 3+ models)
```

### **BenchmarkRunner**

**`run_benchmark(config: ExperimentConfig) → BenchmarkResults`**

```python
results = []
for model_combo in generate_model_combos(config):
    for mode in config.game_modes:
        for seed in config.seeds:
            red_team = build_team(model_combo.red, mode)
            blue_team = build_team(model_combo.blue, mode)
            game_config = GameConfig(mode=mode, seed=seed)
            
            episode = run_episode(game_config, red_team, blue_team)
            metrics = compute_episode_metrics(episode)
            
            results.append(BenchmarkResult(
                model_combo=model_combo,
                mode=mode,
                seed=seed,
                episode=episode,
                metrics=metrics
            ))
            
            save_episode(episode)  # Persist for replay/analysis
            
return aggregate_results(results)
```

### **Leaderboard Schema**

```python
LeaderboardEntry:
    model: str
    role: str                    # "overall", "cluer", "guesser"
    games: int
    win_rate: float
    win_rate_ci: tuple[float, float]  # 95% confidence interval
    avg_coordination_score: float
    avg_clue_efficiency: float   # (cluer only)
    avg_guess_accuracy: float    # (guesser only)
    avg_theory_of_mind: float
```

**Leaderboard views:**
- Overall (by model)
- By role (best cluer, best guesser)
- By mode (who benefits most from discussion?)
- Head-to-head (model A vs model B win rate)

### **Statistical Rigor**

**Sample size:** 100 games per config minimum.

Rationale: For win rate, 100 games gives ±10% margin at 95% confidence. Sufficient for ranking, not for small differences.

**Confidence intervals:** Wilson score interval for proportions (win rate). Standard error for means.

**Paired comparisons:** Same seeds across configs. Enables paired t-test for "does model A beat model B on the same boards?"

**Variance reporting:** Always report std alongside mean. High variance = unreliable metric.

### **Output Artifacts**

```
benchmark_results/
├── {experiment_name}/
│   ├── config.json              # Full experiment config
│   ├── episodes/                # All EpisodeRecords
│   │   ├── episode_001.json
│   │   └── ...
│   ├── metrics/
│   │   ├── per_episode.csv      # One row per episode
│   │   └── aggregate.json       # Summary stats
│   ├── leaderboard.json         # Ranked results
│   └── report.md                # Human-readable summary
```

### **M6 Tests**

**Experiment config:**
- Generates correct number of combinations
- Seeds are reproducible across runs
- Invalid config (e.g., unknown model) raises early

**Benchmark runner:**
- Runs all combinations
- Saves episodes incrementally (crash recovery)
- Handles API failures gracefully (retry with backoff, skip after N failures)
- Progress logging (X of Y complete)

**Leaderboard:**
- Ranks models correctly by win rate
- Confidence intervals computed correctly (manual check)
- Ties handled (same win rate → secondary sort by coordination score)
- Role-specific views filter correctly

**Paired comparisons:**
- Same seed produces same board across configs
- Paired t-test runs without error
- Detects known difference (mock data where A always beats B)

**Output artifacts:**
- All files created in correct structure
- Episodes deserialize correctly
- CSV loadable by pandas
- Markdown renders correctly

**Reproducibility:**
- Re-running same config + seeds produces identical leaderboard (deterministic)
- Or if non-deterministic (temperature > 0), variance is within expected bounds

---

## **Milestone 7 (Stretch): Live Viewer**

**Goal:** Watch games in real-time. Debug, demo, qualitative analysis.

### **Components**

**EventStream:**
- WebSocket or Server-Sent Events
- Pushes transcript events as they occur
- Includes delay parameter (e.g., 2s between events for watchability)

**ViewerState:**
```python
ViewerState:
    board: list[Word]           # 25 words with positions
    revealed: dict[Word, CardType]
    current_turn: Team
    current_phase: Phase
    current_clue: Clue | None
    discussion: list[DiscussionMessage]
    transcript: list[Event]     # Full history
```

**Board Visualization:**
```
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│  APPLE  │  BANK   │ ██DOG██ │  SPRING │  LEAD   │
│         │         │  (RED)  │         │         │
├─────────┼─────────┼─────────┼─────────┼─────────┤
│  PALM   │ ██SUN██ │  MATCH  │  COLD   │  NAIL   │
│         │ (BLUE)  │         │         │         │
├─────────┼─────────┼─────────┼─────────┼─────────┤
...
```

- Unrevealed: plain text
- Revealed: highlighted with team color
- Assassin: skull emoji or distinct marker

**UI Panels:**
1. **Board** — 5x5 grid, updates on reveal
2. **Current Turn** — Team, phase, clue if active
3. **Discussion** — Live stream of guesser messages
4. **Transcript** — Scrollable history
5. **Score** — Words remaining per team

### **Modes**

**Live:** Watch game as it plays. Configurable delay between events.

**Replay:** Load EpisodeRecord, step through events. Forward/back controls.

**Slow-mo:** 5-10s delays for demo/presentation.

### **Optional: Human Takeover**

- Pause game at any point
- Human submits clue/guess instead of agent
- Resume with agents
- Useful for: "What would YOU have done here?"

### **Tech Options**

- **Terminal (Rich):** Fast to build, works over SSH, no browser needed
- **Web (Streamlit):** Richer UI, easier sharing, more setup
- **Web (custom):** Most flexible, most work

Recommendation: Start with Rich terminal viewer. Add web later if needed for demos.

### **M7 Tests**

**Event stream:**
- Events arrive in order
- Delay parameter respected
- Handles slow consumers (backpressure or drop)

**Viewer state:**
- Updates correctly on each event type
- Board reveals update grid
- Discussion clears on new turn

**Replay:**
- Loads episode correctly
- Step forward works
- Step backward works (requires state snapshots or recompute)
- Jump to turn N works

**Visualization:**
- Board renders 5x5 correctly
- Colors/markers distinguish card types
- Handles long words (truncation or wrapping)

---

## **Deferred (Post-M7)**

| Feature | Notes |
|---------|-------|
| Blind mode | Cluers infer key from game flow. Config toggle. |
| Stemming validation | "BANK" invalid if "BANKING" on board. Requires NLP. |
| Vision mode | Agents see board image, not word list. Multimodal. |
| Prompt optimization | Systematic prompt tuning once baseline established. |
| ELO ratings | Head-to-head ratings. Requires many games. |
| Multi-provider abstraction | Unified API across Claude/GPT/Gemini. Build when needed. |

---

## **Full Milestone Summary**

| Milestone | Goal | Success Criteria |
|-----------|------|------------------|
| M0 | Game engine | `pytest test_engine.py` passes, scripted game completes |
| M1 | Cluer agent | 10 boards, 100% legal clues (with retries) |
| M2 | Guesser agent + discussion | Consensus mechanism works, valid ordered guesses |
| M3 | Turn orchestration | Single team wins solo game >50%, no crashes |
| M4 | Adversarial play | 6 agents, full game, valid winner, no info leakage |
| M5 | Metrics | Can compute coordination score, theory of mind, win rate |
| M6 | Benchmark | 100+ games, leaderboard, reproducible, CI on metrics |
| M7 | Live viewer | Watch game real-time, replay episodes |

---

## **Key Decisions Baked In (Full Spec)**

| Decision | Rationale |
|----------|-----------|
| Ghost team for M3 | Test one team in isolation before adversarial complexity |
| PASS ghost as default | Simplest solo mode—real team just needs to clear words |
| Game modes as config | NO_ASSASSIN and SINGLE_GUESSER are ablations, not core |
| 100 games per config | Statistical power for ranking, not small effects |
| Same seeds across configs | Enables paired comparison |
| Coordination score composite | Single number for leaderboard, decomposable for analysis |
| Theory of mind from REASONING | Requires structured output, graceful fallback if missing |
| Terminal viewer first | Fast to build, sufficient for debugging, web is stretch |

---

That's the full unified spec, M0-M7. Coherent with M0-M2's architecture, same level of rigor. Ready for the coding agent.