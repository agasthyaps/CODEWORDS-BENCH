---

## **Codenames Benchmark v1.1: Experimental Design Tightening**

### **Overview**

This update addresses three methodological issues identified in v1.0: side advantage confound, homogeneous team composition, and non-diagnostic ToM metric. Also adds model farm for broader coverage beyond two pinned models.

These fixes establish measurement infrastructure we'll reuse for **Decrypto** (next benchmark). Decrypto has symmetric teams and explicit opponent-modeling baked into game rules, making it a cleaner probe for adversarial ToM. Getting Codenames metrics right first means we can port the ToM scoring and mixed-team analysis directly.

---

### **1. Model Farm**

**Problem:** v1.0 hardcoded Claude and GPT. We want to test across a broader set of models available via OpenRouter.

**Implementation:**

Create `config/models.json`:
```json
{
  "model_farm": [
    {"id": "anthropic/claude-3.5-sonnet", "short_name": "claude-3.5-sonnet"},
    {"id": "openai/gpt-4o", "short_name": "gpt-4o"},
    {"id": "google/gemini-pro-1.5", "short_name": "gemini-1.5"},
    {"id": "meta-llama/llama-3.1-405b-instruct", "short_name": "llama-405b"},
    {"id": "mistral/mistral-large", "short_name": "mistral-large"}
  ],
  "default_matchups": "round_robin",
  "openrouter_base_url": "https://openrouter.ai/api/v1"
}
```

**Matchup generation:**
- `round_robin`: Every model plays every other model (both directions)
- `subset`: Specify explicit matchup list
- Agent reads from config, not hardcoded model strings

**All API calls route through OpenRouter** — no provider-specific clients.

---

### **2. Side Advantage Correction**

**Problem:** RED wins 63% of games across all conditions. Head-to-head matchups only ran one direction, making results uninterpretable.

**Fix:** Every matchup runs **both directions** on the same board seed.

```
For each board seed:
  - Game A: Model X as RED, Model Y as BLUE
  - Game B: Model X as BLUE, Model Y as RED (same board)
```

**Reporting changes:**
- Raw win rates (as before)
- **Side-adjusted win rate**: Average performance across both sides
- **Side advantage delta**: RED win rate − BLUE win rate (per model and overall)

---

### **3. Mixed Team Compositions**

**Problem:** Homogeneous teams conflate cluer skill with guesser skill. Can't determine if a model is better at giving clues vs interpreting them.

**Fix:** Add mixed-team configurations. Minimal version — swap cluer while holding guesser composition constant.

**New configurations:**

For any two models A and B:

| Config | RED Cluer | RED Guessers | BLUE Cluer | BLUE Guessers |
|--------|-----------|--------------|------------|---------------|
| `homog-A` | A | A, A | B | B, B |
| `homog-B` | B | B, B | A | A, A |
| `mixed-A-clue` | A | B, B | B | A, A |
| `mixed-B-clue` | B | A, A | A | B, B |

Each config runs both directions (side swap), so 4 configs × 2 directions = **8 games per model pair per seed**.

**Reporting changes:**
- **Cluer effectiveness**: Win rate when model X is cluer (across all guesser compositions)
- **Guesser effectiveness**: Win rate when model X is guesser (across all cluer compositions)
- **Cluer-guesser synergy**: Does same-model cluer+guessers outperform predicted from independent scores?

---

### **4. Theory of Mind Metric Overhaul**

**Problem:** Current ToM metric is identical to clue efficiency — measures outcomes, not mental modeling.

**Fix:** Implement **prediction-based ToM scoring**. Cluer predicts teammate behavior *after* committing to clue, before seeing results.

**Cluer prediction prompt** (after clue delivered, before guesser discussion):

```
You gave the clue: [CLUE, NUMBER]

Before seeing your teammates' discussion:

1. PREDICTED_GUESSES: What words do you think your teammates will guess, in order?
2. CONFUSION_RISKS: Which non-target words might they incorrectly consider? (list with brief reason)
```

**Stored as:**
```json
{
  "clue": "RIVER, 2",
  "intended_targets": ["BANK", "STREAM"],
  "predicted_guesses": ["BANK", "STREAM", "FLOW"],
  "confusion_risks": [
    {"word": "MONEY", "reason": "BANK polysemy"},
    {"word": "FISH", "reason": "river association"}
  ]
}
```

Note: `intended_targets` captured at clue selection. Prediction prompt comes after delivery — no revision opportunity.

**ToM scoring (per clue):**
```
prediction_accuracy = |predicted_guesses ∩ actual_guesses| / |actual_guesses|
rank_correlation = Spearman correlation between predicted order and actual guess order
confusion_calibration = |confusion_risks ∩ actual_wrong_guesses| / |actual_wrong_guesses|
```

**Aggregate (per model as cluer):**
```
team_tom = mean(prediction_accuracy) across all clues
```

---

### **5. Updated Output Schema**

```json
{
  "game_id": "...",
  "board_seed": "...",
  "config": {
    "red_cluer": "claude-3.5-sonnet",
    "red_guessers": ["gpt-4o", "gpt-4o"],
    "blue_cluer": "gpt-4o", 
    "blue_guessers": ["claude-3.5-sonnet", "claude-3.5-sonnet"],
    "config_type": "mixed-A-clue"
  },
  "result": {
    "winner": "RED",
    "rounds": 4,
    "assassin_hit": false
  },
  "clue_log": [
    {
      "round": 1,
      "team": "RED",
      "clue": "RIVER, 2",
      "intended_targets": ["BANK", "STREAM"],
      "predicted_guesses": ["BANK", "STREAM"],
      "confusion_risks": [{"word": "MONEY", "reason": "..."}],
      "actual_guesses": ["BANK", "STREAM"],
      "prediction_accuracy": 1.0
    }
  ],
  "tom_scores": {
    "red_cluer_team_tom": 0.73,
    "blue_cluer_team_tom": 0.58
  },
  "side_info": {
    "direction": "A_RED_B_BLUE",
    "paired_game_id": "..." 
  }
}
```

---

### **6. Run Matrix**

For N models in farm, round-robin generates N×(N-1)/2 unique pairs.

Per pair:
- 4 configs (2 homogeneous + 2 mixed)
- 2 directions each
- S seeds

**Total games = N×(N-1)/2 × 4 × 2 × S**

Example with 5 models, 5 seeds:
- 10 pairs × 4 configs × 2 directions × 5 seeds = **400 games**

Recommend starting with 3 models, 5 seeds = **120 games** for validation run.

---

### **7. Analysis Outputs**

1. **Side advantage analysis**
   - Overall RED vs BLUE win rate
   - Per-model side advantage delta

2. **Model rankings (side-adjusted)**
   - Overall win rate
   - Cluer effectiveness (when model is cluer)
   - Guesser effectiveness (when model is guesser)

3. **ToM analysis**
   - Mean Team-ToM by model
   - Correlation: ToM score ↔ win rate
   - Confusion calibration scores

4. **Synergy analysis**
   - Homogeneous vs mixed team performance
   - Best cross-model pairings

---

### **8. Future Work: Decrypto**

This benchmark establishes infrastructure for **Decrypto**, which adds:

- **Symmetric teams**: No side advantage by construction
- **Adversarial ToM**: You're explicitly trying to crack opponent's code while protecting your own
- **Opponent-ToM scoring**: Predict what opponents will infer from your clues (leakage metric)
- **Discussion ablation**: ON/OFF conditions for team/opponent visibility

The model farm, mixed-team analysis, and ToM prediction framework port directly. Decrypto's game structure makes "using opponent discussion" measurable by design — the Meta/Oxford paper (Lupu et al., June 2025) validated this construct, though with single-team setup. Our contribution is full two-team adversarial play.
