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


# Hardcoded context about the benchmark - concise and focused
_BENCHMARK_CONTEXT = """
# Codewords Benchmark Overview

This is an LLM coordination benchmark using word games (Codenames & Decrypto) to evaluate multi-agent communication.

## Codenames
- 5x5 board with words assigned to RED (9), BLUE (8), NEUTRAL (7), ASSASSIN (1)
- Teams have a Cluer (gives one-word clues + number) and Guessers (discuss, then guess words)
- Win by revealing all your team's words; lose by hitting assassin or letting opponent win
- Key metrics: clue efficiency (words per clue), guess precision, coordination rate

## Decrypto  
- Each team has 4 secret keywords (positions 1-4)
- Each round: Cluer gives 3 clues for a 3-digit code (e.g., "forest, metal, luck" for 1-3-4)
- Own team must DECODE (guess own code); opponent tries to INTERCEPT (guess opponent's code)
- Win by 2 interceptions OR opponent gets 2 miscommunications
- Key metrics: decode accuracy, intercept rate, clue leakage (how much opponents learn)

## Metrics to Analyze
- **consensus_rate**: Did guessers reach explicit agreement? (Look for "CONSENSUS: YES" in transcripts)
- **theory_of_mind**: Does cluer predict teammate reasoning? (Currently requires tom_predictions in output)
- **calibration**: Are confidence estimates well-calibrated to outcomes?
- **clue_efficiency**: How many words correctly guessed per clue?
- **intercept_rate**: For Decrypto, how often does opponent guess the code?

## Known Issues
- consensus_rate may show 0% even when agents agreed — check if parser recognizes the format
- ToM metrics require explicit prediction prompts which may not be implemented
- Metrics marked "null" indicate missing data, not necessarily poor performance

Your analysis should help make sense of the data and identify patterns and trends.
"""


async def analyze_and_save(
    *,
    game_type: GameType,
    replay_id: str,
    episode: dict[str, Any],
) -> dict[str, Any]:
    payload = _build_payload(game_type, episode)

    system_prompt = (
        "You are a research analyst for the Codewords/Decrypto LLM benchmark. "
        "Analyze the game results and metrics to evaluate coordination quality, "
        "communication efficiency, and strategic reasoning.\n\n"
        "Guidelines:\n"
        "- Be concrete: cite specific metrics, notable plays, and failure modes\n"
        "- Note any metrics that seem like measurement artifacts (e.g., 0% when agents clearly agreed)\n"
        "- Suggest specific improvements for the agents or benchmark\n"
        "- Keep analysis focused and actionable (3-5 key observations)\n\n"
        f"{_BENCHMARK_CONTEXT}"
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
