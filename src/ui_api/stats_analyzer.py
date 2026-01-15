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

_SPEC_CACHE: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_spec_context() -> str:
    global _SPEC_CACHE
    if _SPEC_CACHE is not None:
        return _SPEC_CACHE
    spec_files = [
        _repo_root() / "spec_archive" / "spec_m0-m2.md",
        _repo_root() / "spec_archive" / "spec_m3-m7.md",
        _repo_root() / "spec_archive" / "decrypto_spec.md",
        _repo_root() / "spec_archive" / "improvements.md",
    ]
    chunks: list[str] = []
    for path in spec_files:
        if not path.exists():
            continue
        with open(path, "r") as f:
            chunks.append(f"\n\n# {path.name}\n{f.read()}")
    _SPEC_CACHE = "\n".join(chunks)
    return _SPEC_CACHE


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


async def analyze_and_save(
    *,
    game_type: GameType,
    replay_id: str,
    episode: dict[str, Any],
) -> dict[str, Any]:
    context = _load_spec_context()
    payload = _build_payload(game_type, episode)

    system_prompt = (
        "You are a research analyst for the Codewords/Decrypto benchmark. "
        "Explain what the stats imply about coordination quality, strategy, and failure modes. "
        "Compare to the project's goals and metrics definitions from the spec context. "
        "Be concrete: cite key metrics, standout turns, and suggest follow-up experiments."
    )
    user_prompt = json.dumps(payload, indent=2)

    provider = OpenRouterProvider(
        model="anthropic/claude-opus-4.5",
        base_url=_openrouter_base_url() or "https://openrouter.ai/api/v1",
    )
    response = await provider.complete(
        messages=[
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
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
