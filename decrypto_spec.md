## **Decrypto Benchmark: Full Adversarial Play (Tightened Spec v0.2)**

### **Overview**

Decrypto is an asymmetric-information word game where two teams simultaneously communicate secret codes while intercepting the opponent’s codes. Unlike the Meta/Oxford benchmark (Lupu et al., June 2025) which uses a simplified 3-player setup (encoder, decoder, interceptor), this benchmark implements **full two-team competitive play**.

**Core contribution:** Full symmetric play forces each model to encode *and* intercept under compounding strategic pressure across rounds. This benchmark asks not just whether models can “mindread,” but whether they can **maintain a private semantic code under adversarial observation** and **adapt policy under competitive state pressure**.

This benchmark shares infrastructure with Codenames (model farm, experiment harness, deliberation protocol, ToM prompt scaffolds) and is designed for cross-game comparison.

---

## 1. Game Rules

**Setup:**

* Two teams (RED, BLUE), each with 3 agents: 1 cluer (pinned), 2 guessers
* Each team has a secret 4-word key, hidden from opponents
* Game lasts up to 8 rounds

**Each round, both teams play:**

1. **Clue phase:** Each cluer receives a secret 3-digit code (e.g., 2-4-1) referencing positions in their key. Cluer gives 3 clues (one per digit, in order).
2. **Intercept phase:** Opposing guessers confer and guess the code based on current clues + history. (They do *not* know the key.)
3. **Decode phase:** Own guessers confer and guess their own code.
4. **Reveal:** True code, both guesses revealed. History updated.

**Win conditions:**

* **Win by interception:** First team to correctly guess opponent's code twice
* **Lose by miscommunication:** First team whose own guessers fail twice
* **Survive:** 8 rounds without either condition

**Information visibility:**

| Information                 | Own Cluer | Own Guessers | Opponent |
| --------------------------- | --------: | -----------: | -------: |
| Own 4-word key              |         ✅ |            ✅ |        ❌ |
| Own current code            |         ✅ |            ❌ |        ❌ |
| All past clues (both teams) |         ✅ |            ✅ |        ✅ |
| All past codes (both teams) |         ✅ |            ✅ |        ✅ |
| All past final guesses      |         ✅ |            ✅ |        ✅ |

**Note:** Deliberations are private. The public history includes clues, true codes, and final guesses only.

**Code generation:**

* 3 distinct digits from {1,2,3,4}, randomly permuted
* Each of 24 possible codes used at most once per game

**Keyword bank:** 680 words (from Meta paper’s curated set)

---

## 2. Model Farm

Same setup as Codenames v1.1. All API calls via OpenRouter.

`config/models.json`:

```json
{
  "model_farm": [
    {"id": "anthropic/claude-3.5-sonnet", "short_name": "claude-3.5-sonnet"},
    {"id": "openai/gpt-4o", "short_name": "gpt-4o"},
    {"id": "google/gemini-pro-1.5", "short_name": "gemini-1.5"},
    {"id": "meta-llama/llama-3.1-405b-instruct", "short_name": "llama-405b"}
  ],
  "default_matchups": "round_robin",
  "openrouter_base_url": "https://openrouter.ai/api/v1"
}
```

---

## 3. Team Compositions

**No side counterbalancing needed** — Decrypto is symmetric.

For any two models A and B:

| Config         | RED Cluer | RED Guessers | BLUE Cluer | BLUE Guessers |
| -------------- | --------- | ------------ | ---------- | ------------- |
| `homog-A`      | A         | A, A         | B          | B, B          |
| `homog-B`      | B         | B, B         | A          | A, A          |
| `mixed-A-clue` | A         | B, B         | B          | A, A          |
| `mixed-B-clue` | B         | A, A         | A          | B, B          |

**4 configs per model pair.**

---

## 4. Agent Roles

### 4.1 Cluer

**Input:**

```json
{
  "role": "cluer",
  "team": "red",
  "key": ["WHALE", "CLOCK", "FOREST", "PIANO"],
  "code": [2, 4, 1],
  "history": {
    "own": [{"round": 1, "code": [3,1,2], "clues": ["...", "...", "..."], "intercepted": false, "team_correct": true}],
    "opponent": [{"round": 1, "code": [1,4,3], "clues": ["...", "...", "..."], "intercepted": false, "team_correct": true}]
  },
  "game_state": {"own_interceptions": 0, "own_miscommunications": 0, "opp_interceptions": 1, "opp_miscommunications": 0}
}
```

**Output:**

```json
{
  "clues": ["TICK", "KEYS", "OCEAN"],
  "annotations": {
    "intended_mapping": {"2": "CLOCK", "4": "PIANO", "1": "WHALE"},
    "clue_rationale": {"TICK": "CLOCK", "KEYS": "PIANO", "OCEAN": "WHALE"},
    "risk_estimates": {
      "predicted_team_guess": [2,4,1],
      "predicted_team_confidence": 0.85,
      "predicted_intercept_probability": 0.20
    }
  }
}
```

**Constraints:**

* Exactly 3 clues, one per code position, in order
* Single words only (or short phrases per original rules)
* Cannot use key words verbatim

**Note:** the `annotations` are *for evaluation only* and are hidden from other agents.

---

### 4.2 Guessers: Own Decode

**Input:** key + current clues + full history

**Output:** consensus code guess via deliberation mechanism.

**Deliberation logging requirement (tightening):**

* Log each guesser’s *initial independent guess* before discussion
* Log final consensus guess
* Log whether/where agents revised

This separates:

* epistemic accuracy (“what did they believe?”)
* coordination dynamics (“how did they converge?”)

---

### 4.3 Guessers: Opponent Intercept

**Input:** own key + current clues + full history (opponent key hidden)

**Output:** consensus intercept guess

**Additional logging (tightening):**

* Each guesser provides:

  * independent intercept guess
  * confidence (0–1)
  * inferred mapping distribution over positions (soft, optional)

---

## 5. Theory of Mind Metrics

### A. Team-ToM (same as Codenames)

After cluer commits to clues, before own team guesses:

> Predict what code your teammates will guess.

**Scoring:**

* `team_tom = accuracy(predicted_team_guess == actual_team_guess)` aggregated across rounds

Also compute:

* `team_calibration = correlation(predicted_team_confidence, team_correct)`

---

### B. Opponent-ToM (new for Decrypto)

After cluer commits to clues, before opponent guesses:

> Predict whether the opponent will intercept (guess correctly), and estimate probability.

**Scoring:**

* `opponent_tom = accuracy( (p_intercept>0.5) == intercept_happened )`
* `leakage_awareness = correlation(p_intercept, intercept_happened)` (or AUROC)

Interpretation tightening:

* This measures *awareness of leakage risk*,
* not necessarily ability to prevent leakage.

---

### C. Intercept-ToM (when guessing opponent's code)

Before submitting intercept guess:

* confidence (0–1)
* inferred mapping of opponent’s positions 1–4 to latent “themes” or key guesses (optional)

**Scoring:**

* `intercept_calibration = correlation(confidence, correct_intercept)`
* If key inference is enabled (requires postgame reveal), compute:

  * `key_inference_accuracy` (position-wise match rate or top-k accuracy)

---

## 6. Strategic Adaptation Metrics (Novel Contribution)

### 6.1 State-conditioned performance

Define round-state at time of clue generation:

* **Leading:** own_interceptions > opp_interceptions
* **Trailing:** own_interceptions < opp_interceptions
* **Tied:** equal interceptions
* **Danger:** own_miscommunications == 1 (one more = loss)

**Metrics by state:**

* decode success rate
* intercepted-against rate
* time-to-consensus (proxy: # deliberation turns)
* clue strategy changes (see 6.2)

---

### 6.2 Clue strategy: split “diversity” into two axes (tightening)

#### A. Surface Diversity

Measures whether clue *wording* changes:

* embedding distance between clues for the same key position across rounds
* lexical novelty rate

#### B. Latent Mapping Stability

Measures whether the cluer maintains a consistent semantic code:

* similarity between the cluer’s `intended_mapping` rationale across rounds (position-wise)
* or an embedding-based consistency score between clue rationales over time

**Why this matters:**
A model that “thrashes” with new words but changes meaning isn’t strategically diverse—it’s incoherent.

**Desired pattern under pressure:**
high surface diversity + high latent mapping stability.

---

### 6.3 Risk-taking vs safety (operationalized)

Instead of a vibes-based “clever vs safe,” define proxies:

* **Polysemy / ambiguity proxy:** clue embedding dispersion vs multiple key words
* **Concreteness proxy:** use a concreteness lexicon when available (optional)
* **Directness proxy:** similarity(clue, key word) in embedding space (too close = “safe but leaky”)
* **Obfuscation proxy:** similarity(clue, opponent-inferred theme clusters) decreases after interception

---

### 6.4 Cross-round learning after being intercepted

After an interception **against** a team:

* Does next-round `p_intercept` decrease? (risk awareness)
* Does surface similarity to previously cracked clue decrease?
* Does latent mapping stability remain high while surface shifts?

This captures whether models can “rotate the cipher” without losing their own teammates.

---

## 7. Output Schema (Updated)

Key additions:

* independent guesses before deliberation
* cluer annotations (hidden from other agents)
* split diversity metrics

```json
{
  "game_id": "...",
  "seed": "...",
  "config": {...},
  "keys": {...},
  "rounds": [
    {
      "round": 1,
      "red_turn": {
        "code": [2,4,1],
        "clues": ["TICK","KEYS","OCEAN"],
        "cluer_annotations": {
          "intended_mapping": {"2":"CLOCK","4":"PIANO","1":"WHALE"},
          "risk": {"p_team_correct": 0.85, "p_intercept": 0.20}
        },
        "team_decode": {
          "guesser_independent": [
            {"agent":"red_g1","guess":[2,4,1],"confidence":0.8},
            {"agent":"red_g2","guess":[2,4,1],"confidence":0.7}
          ],
          "deliberation": [{"speaker":"red_g1","text":"..."},{"speaker":"red_g2","text":"..."}],
          "final_guess": [2,4,1],
          "team_correct": true
        },
        "opponent_intercept": {
          "guesser_independent": [
            {"agent":"blue_i1","guess":[2,3,1],"confidence":0.5},
            {"agent":"blue_i2","guess":[2,4,1],"confidence":0.4}
          ],
          "final_guess": [2,3,1],
          "intercept_correct": false
        }
      },
      "blue_turn": { "...": "..." }
    }
  ],
  "result": {...},
  "scores": {
    "tom": {...},
    "calibration": {...},
    "adaptation": {
      "surface_diversity_by_round": {"red":[null,0.7,0.8], "blue":[null,0.5,0.6]},
      "latent_mapping_stability_by_round": {"red":[null,0.9,0.85], "blue":[null,0.8,0.7]},
      "interception_rate_by_round": [0.0,0.0,0.5,1.0]
    }
  }
}
```

---

## 8. Run Matrix

For N models:

* pairs = N×(N−1)/2
* configs per pair = 4
* seeds = S

**Total games = N×(N−1)/2 × 4 × S**

Example with 4 models, 5 seeds:

* 6 pairs × 4 configs × 5 seeds = **120 games**

---

## 9. Analysis Outputs (Tightened Interpretability)

1. **Outcome decomposition**

   * win by interception vs win by opponent miscommunication
   * average game length by matchup

2. **Role-specific strength**

   * cluer: protect vs leak tradeoff (own decode accuracy vs intercepted-against rate)
   * guesser: own decode accuracy + time-to-consensus + revision rate
   * interceptor: intercept accuracy + calibration

3. **ToM and calibration**

   * team_tom and team_calibration by model
   * opponent_tom and leakage_awareness by model
   * correlate with role performance, not just win rate

4. **Adaptation**

   * state-conditioned changes in:

     * surface diversity
     * latent mapping stability
     * risk estimates (p_intercept)
   * post-interception “cipher rotation” behavior:

     * surface shifts without latent collapse

5. **Cross-game comparison (Codenames)**

   * skill transfer vs divergence:

     * deliberation stability
     * leakage management
     * opponent modeling

---

## 10. Implementation Notes

**Shared with Codenames:**

* Model farm / OpenRouter client
* Consensus mechanism scaffolding (with independent guess logging added)
* Experiment harness (matchup generation, seeding, logging)
* ToM prompt scaffolds

**New for Decrypto:**

* Symmetric dual-turn round orchestrator
* Intercept phase orchestration and visibility
* History structure
* Keyword bank (Meta’s curated 680)

**Milestones:**

| Milestone | Deliverable                                         |
| --------- | --------------------------------------------------- |
| M0        | Game state machine + validation + phase transitions |
| M1        | Cluer agent + hidden annotations logging            |
| M2        | Guesser agents + independent guesses + consensus    |
| M3        | Full dual-team round orchestration                  |
| M4        | Full games + complete logging schema                |
| M5        | Experiment harness + baseline metrics               |
| M6        | ToM + calibration + adaptation analyses             |

---

## 11. Relationship to Prior Work (Sharper Positioning)

**Meta/Oxford (Lupu et al., June 2025):**

* asymmetric 3-player setup
* ToM measured via task analogies (Smarties, perspective-taking)
* key claim: reasoning models underperform on ToM proxies

**This benchmark extends the construct:**

* not “ToM in isolation,” but **adversarial meaning maintenance**
* models must sustain a private code while:

  * coordinating internally
  * being observed externally
  * adapting under shifting win/loss conditions

**Research questions enabled here:**

* Do models change *policy* under pressure (trailing/danger), not just tactics?
* Can models rotate surface clues while preserving latent mapping?
* How much do deliberation dynamics (revision vs dominance) affect robustness?
* Is “being good at communication” inherently leaky under observation?
