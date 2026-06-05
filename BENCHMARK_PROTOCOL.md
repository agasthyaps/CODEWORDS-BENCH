# Humanlike Game Benchmark Protocol

## Research Goal

This platform measures how language-model agents play cooperative and team games under humanlike table conditions. The goal is observational: preserve the agent's raw behavior, record where the harness intervenes, and avoid optimizing the harness to maximize game score.

## Default Condition

The default condition is `human_table_scratchpad`.

- Private scratchpads are enabled and persist across turns within an episode.
- Neutral format and legality repair is allowed.
- Invalid actions are handled with a human-table policy when available: record the illegal proposal, apply a moderator correction that keeps play moving, and penalize benchmark performance.
- Harness interventions are logged separately from game score.

Diagnostic conditions:

- `raw_chat`: no persistent scratchpad and no format repair beyond transport-level provider retries.
- `structured_output`: provider/schema-enforced output as a comparator, not the default raw-agent condition.

## Game Rules And Visibility

Each game keeps its canonical hidden-information boundary:

- Codenames cluers see the key; guessers see only public board state and transcript.
- Decrypto cluers see their team's key and current code; guessers see only role-appropriate public/private views.
- Hanabi players see other players' hands, table state, hints, and their own card knowledge, but not their own cards.

## Model Matrix

Use homogeneous baselines first:

- One primary representative per model family: OpenAI, Anthropic, Google, and optionally one open-weight model.
- Same-model teams across Codenames, Decrypto, and Hanabi.
- Full seed set for the main comparison surface.

Use sparse mixed-family cross-play second:

- Pair family representatives rather than every model variant.
- Include OpenAI-Anthropic, OpenAI-Google, Anthropic-Google, and selected open-weight pairings.
- Counterbalance roles for asymmetric games.
- Use fewer seeds and report mixed runs as coordination analysis, not a total leaderboard.

Seed policies are versioned in code:

- Pilot homogeneous: seeds `0..4`.
- Pilot mixed: seeds `0..2`.
- Main homogeneous: seeds `0..29`.
- Main mixed: seeds `0..9`.

## Run Manifest

Every benchmark artifact should include a machine-readable `run_manifest` with:

- condition name and policies
- game type and rules version
- model metadata and provider parameters
- seed schedule
- prompt hashes
- git SHA and dirty-worktree flag when available

## Penalties And Interpretation

Game outcome and benchmark performance are separate.

Game outcome includes normal game score, winner, turns, rounds, and terminal reason.

Benchmark performance includes:

- parse failures
- repair prompts
- fallback actions
- invalid actions
- moderator corrections
- transport retries when available

A model can win a game while still showing high harness dependence. Reports should foreground that distinction.

## Known Limitations

- `structured_output` is a named comparator condition; provider-native schema calls may need provider-specific implementation before use.
- Some agents still include scratchpad instructions in prompt templates even when `raw_chat` disables persistence.
- LLM qualitative analysis should be treated as annotation unless it cites transcript evidence.
