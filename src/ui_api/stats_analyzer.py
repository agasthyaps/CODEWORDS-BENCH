"""Stats analyzer using OpenRouter (Opus 4.5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.agents.llm import OpenRouterProvider
from src.metrics import compute_episode_metrics
from src.metrics.models import EpisodeMetrics
from src.runner.episode import ExtendedEpisodeRecord

from .storage import save_stats_report

GameType = Literal["codenames", "decrypto"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _openrouter_base_url() -> str | None:
    config_path = _repo_root() / "config" / "models.json"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        data = json.load(f)
    return data.get("openrouter_base_url")


def _build_payload(game_type: GameType, episode: dict[str, Any]) -> dict[str, Any]:
    if game_type == "codenames":
        parsed = ExtendedEpisodeRecord.model_validate(episode)
        metrics: EpisodeMetrics = compute_episode_metrics(parsed)
        return {
            "game_type": "codenames",
            "metrics": metrics.model_dump(mode="json"),
            "winner": episode.get("winner"),
            "turns": episode.get("total_turns"),
            "team_metadata": episode.get("metadata", {}),
            "episode": episode,
        }
    return {
        "game_type": "decrypto",
        "scores": episode.get("scores", {}),
        "winner": episode.get("winner"),
        "result_reason": episode.get("result_reason"),
        "episode": episode,
    }


# Context about the research platform - focused on understanding model behavior
_RESEARCH_CONTEXT = """
# Model Coordination Research Platform

This is NOT a benchmark for optimizing game performance. Instead, it's a research platform 
designed to observe and understand how language models naturally coordinate, communicate, 
and reason about each other's mental states in games with asymmetric information.

## Research Goals
- **Latent Theory of Mind**: Do models spontaneously reason about what teammates know/believe?
- **Coordination Patterns**: How do models establish shared understanding without explicit protocols?
- **Communication Strategies**: What linguistic patterns emerge in model-to-model communication?
- **Failure Modes**: Where does coordination break down, and why?

We provide minimal scaffolding intentionally — the goal is to reveal what's latent in the models,
not to optimize their play through better prompts or structures.

## Games as Research Instruments

### Codenames
- 5x5 board: RED (9), BLUE (8), NEUTRAL (7), ASSASSIN (1) words
- Cluer gives one-word clue + number; Guessers discuss then guess
- **What to observe**: How does the cluer craft clues anticipating teammate interpretation?
  Do guessers reason about the cluer's perspective? How do they resolve ambiguity through discussion?

### Decrypto  
- Each team has 4 secret keywords at positions 1-4
- Cluer gives 3 clues mapping to a 3-digit code (e.g., "forest, metal, luck" → 1-3-4)
- Teams decode own codes; opponents attempt interception
- **What to observe**: How do cluers balance being understood by teammates vs. not being intercepted?
  Do models track patterns across rounds? How do they update beliefs about opponent keywords?

## What to Look For
- **Spontaneous perspective-taking**: Does the cluer mention considering what guessers might think?
- **Belief tracking**: Do guessers reason about what information the cluer has access to?
- **Discussion dynamics**: How do two guessers converge (or fail to) on a shared interpretation?
- **Adaptation over time**: Do models learn from previous rounds/turns?
- **Scratchpad usage**: If agents use their private scratchpad, what do they choose to record?

## Analysis Focus
Rather than evaluating "how well" models played, focus on:
1. **Coordination phenomena**: Interesting patterns in how models work together
2. **Theory of Mind signals**: Evidence of reasoning about others' knowledge/beliefs
3. **Communication strategies**: Notable approaches to conveying or interpreting information
4. **Breakdowns**: Where coordination failed and what that reveals about model limitations

Note: Metrics like "efficiency" or "win rate" are secondary — we care more about the 
qualitative nature of model interactions than optimizing outcomes.
"""


async def analyze_and_save(
    *,
    game_type: GameType,
    replay_id: str,
    episode: dict[str, Any],
) -> dict[str, Any]:
    payload = _build_payload(game_type, episode)

    system_prompt = (
        "You are a cognitive science researcher studying multi-agent coordination and "
        "Theory of Mind in language models. You're analyzing game transcripts to understand "
        "how models naturally coordinate and reason about each other.\n\n"
        "Guidelines:\n"
        "- Focus on BEHAVIOR over PERFORMANCE: we care about how models think, not how well they score\n"
        "- Look for Theory of Mind signals: perspective-taking, belief reasoning, anticipation\n"
        "- Note interesting coordination patterns or communication strategies\n"
        "- Identify breakdowns: where did understanding fail and what does that reveal?\n"
        "- Quote specific transcript excerpts that illustrate your observations\n"
        "- Keep analysis substantive but focused (3-5 key observations)\n\n"
        f"{_RESEARCH_CONTEXT}"
    )
    user_prompt = json.dumps(payload, indent=2)

    provider = OpenRouterProvider(
        model="anthropic/claude-opus-4.5",
        base_url=_openrouter_base_url() or "https://openrouter.ai/api/v1",
    )
    response = await provider.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    report = {
        "replay_id": replay_id,
        "game_type": game_type,
        "analysis": response.content,
        "model": response.model,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        },
    }
    save_stats_report(replay_id, report)
    return report
