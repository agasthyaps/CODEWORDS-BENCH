"""Interim analysis for benchmark batches."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agents.llm import OpenRouterProvider


class InterimFinding(BaseModel):
    """A finding from interim analysis of a batch of games."""

    finding_id: str
    game_type: Literal["codenames", "decrypto", "hanabi"]
    batch_number: int
    games_analyzed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    summary_metrics: dict[str, Any]
    analysis: str
    model: str = "anthropic/claude-opus-4.5"
    usage: dict[str, int] | None = None


def _build_codenames_summary(results: list[dict]) -> dict[str, Any]:
    """Build summary metrics for Codenames games."""
    total = len(results)
    if total == 0:
        return {}

    wins = {"red": 0, "blue": 0, "draw": 0}
    total_turns = []
    models_seen = set()

    for r in results:
        winner = r.get("winner")
        if winner == "RED":
            wins["red"] += 1
        elif winner == "BLUE":
            wins["blue"] += 1
        else:
            wins["draw"] += 1

        if r.get("metrics", {}).get("turns_to_win"):
            total_turns.append(r["metrics"]["turns_to_win"])

        for role in ["cluer", "guesser_1", "guesser_2"]:
            if r.get("red_models", {}).get(role):
                models_seen.add(r["red_models"][role])
            if r.get("blue_models", {}).get(role):
                models_seen.add(r["blue_models"][role])

    return {
        "total_games": total,
        "win_rates": {
            "red": wins["red"] / total,
            "blue": wins["blue"] / total,
            "draw": wins["draw"] / total,
        },
        "avg_turns": sum(total_turns) / len(total_turns) if total_turns else None,
        "models_tested": list(models_seen),
    }


def _build_decrypto_summary(results: list[dict]) -> dict[str, Any]:
    """Build summary metrics for Decrypto games."""
    total = len(results)
    if total == 0:
        return {}

    wins = {"red": 0, "blue": 0, "draw": 0}
    reasons = {}
    models_seen = set()

    for r in results:
        winner = r.get("winner")
        if winner == "red":
            wins["red"] += 1
        elif winner == "blue":
            wins["blue"] += 1
        else:
            wins["draw"] += 1

        reason = r.get("result_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

        for role in ["cluer", "guesser_1", "guesser_2"]:
            if r.get("red_models", {}).get(role):
                models_seen.add(r["red_models"][role])
            if r.get("blue_models", {}).get(role):
                models_seen.add(r["blue_models"][role])

    return {
        "total_games": total,
        "win_rates": {
            "red": wins["red"] / total,
            "blue": wins["blue"] / total,
            "draw": wins["draw"] / total,
        },
        "result_reasons": reasons,
        "models_tested": list(models_seen),
    }


def _build_hanabi_summary(results: list[dict]) -> dict[str, Any]:
    """Build summary metrics for Hanabi games."""
    total = len(results)
    if total == 0:
        return {}

    scores = []
    reasons = {}
    models_seen = set()

    for r in results:
        score = r.get("score", 0)
        scores.append(score)

        reason = r.get("game_over_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

        if r.get("model"):
            models_seen.add(r["model"])

    return {
        "total_games": total,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "perfect_games": sum(1 for s in scores if s == 25),
        "game_over_reasons": reasons,
        "models_tested": list(models_seen),
    }


def _build_analysis_prompt(
    game_type: Literal["codenames", "decrypto", "hanabi"],
    summary: dict[str, Any],
    results: list[dict],
) -> str:
    """Build the analysis prompt for Opus."""
    game_descriptions = {
        "codenames": (
            "Codenames: A word association game where cluers give one-word clues "
            "and guessers must identify team words while avoiding the assassin."
        ),
        "decrypto": (
            "Decrypto: Teams give clues for 3-digit codes based on secret keywords. "
            "Teams decode their own codes while trying to intercept opponent codes."
        ),
        "hanabi": (
            "Hanabi: A cooperative card game where players can see others' cards "
            "but not their own, and must give hints to build fireworks (score 0-25)."
        ),
    }

    prompt = f"""Analyze this batch of {game_type} games from our model coordination research.

## Game Context
{game_descriptions[game_type]}

## Summary Metrics
{json.dumps(summary, indent=2)}

## Sample Games (first 3)
{json.dumps(results[:3], indent=2, default=str)}

## Analysis Focus
This research studies how language models naturally coordinate and reason about each other's mental states. We're NOT optimizing performance - we're observing behavior patterns.

Please provide:
1. **Coordination Patterns**: What strategies or patterns emerge in how models work together?
2. **Theory of Mind Signals**: Evidence of models reasoning about what others know/believe
3. **Notable Breakdowns**: Where did coordination fail and what does that reveal?
4. **Model Differences**: Any observable differences between models tested (if multiple)?
5. **Key Observations**: 2-3 specific interesting moments from the games

Keep analysis substantive but focused (~500 words). Quote specific examples when relevant.
"""
    return prompt


def _get_openrouter_base_url() -> str:
    """Get OpenRouter base URL from config."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "models.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            data = json.load(f)
        return data.get("openrouter_base_url", "https://openrouter.ai/api/v1")
    return "https://openrouter.ai/api/v1"


async def analyze_batch(
    game_type: Literal["codenames", "decrypto", "hanabi"],
    results: list[dict],
    batch_number: int,
    output_dir: Path,
) -> InterimFinding:
    """
    Analyze a batch of games with Opus.

    Args:
        game_type: Type of game
        results: List of game result dicts
        batch_number: Batch number for this game type
        output_dir: Directory to save findings

    Returns:
        InterimFinding with analysis
    """
    # Build summary
    if game_type == "codenames":
        summary = _build_codenames_summary(results)
    elif game_type == "decrypto":
        summary = _build_decrypto_summary(results)
    else:
        summary = _build_hanabi_summary(results)

    # Build prompt
    prompt = _build_analysis_prompt(game_type, summary, results)

    # Call Opus
    provider = OpenRouterProvider(
        model="anthropic/claude-opus-4.5",
        base_url=_get_openrouter_base_url(),
    )

    system_prompt = (
        "You are a cognitive science researcher studying multi-agent coordination. "
        "Analyze game transcripts to understand how language models coordinate and "
        "reason about each other. Focus on behavior patterns, not performance metrics."
    )

    response = await provider.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    # Create finding
    finding_id = f"{game_type}_{batch_number:04d}"
    finding = InterimFinding(
        finding_id=finding_id,
        game_type=game_type,
        batch_number=batch_number,
        games_analyzed=len(results),
        summary_metrics=summary,
        analysis=response.content,
        model=response.model,
        usage={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        },
    )

    # Save finding
    findings_dir = output_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    finding_path = findings_dir / f"{finding_id}.json"
    with open(finding_path, "w") as f:
        json.dump(finding.model_dump(mode="json"), f, indent=2, default=str)

    return finding


def load_finding(output_dir: Path, finding_id: str) -> InterimFinding | None:
    """Load a finding from disk."""
    path = output_dir / "findings" / f"{finding_id}.json"
    if not path.exists():
        return None

    with open(path, "r") as f:
        data = json.load(f)
    return InterimFinding.model_validate(data)


def list_findings(output_dir: Path) -> list[InterimFinding]:
    """List all findings for an experiment."""
    findings_dir = output_dir / "findings"
    if not findings_dir.exists():
        return []

    findings = []
    for path in sorted(findings_dir.glob("*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        findings.append(InterimFinding.model_validate(data))

    return findings
