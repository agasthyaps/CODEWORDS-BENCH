"""Cost estimation module for games, batches, and benchmarks.

Fetches model pricing from OpenRouter and computes cost estimates based on
historical usage data from episodes.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pydantic import BaseModel


# ============================================
# Pricing Models
# ============================================

@dataclass
class ModelPricing:
    """Pricing for a single model (USD per token/request/unit)."""
    model_id: str
    prompt: float = 0.0  # per token
    completion: float = 0.0  # per token
    request: float = 0.0  # per request
    image: float = 0.0  # per image
    web_search: float = 0.0  # per search
    internal_reasoning: float = 0.0  # per token (for reasoning models)
    input_cache_read: float = 0.0  # per token
    input_cache_write: float = 0.0  # per token

    @classmethod
    def from_openrouter(cls, model_id: str, pricing: dict) -> "ModelPricing":
        """Create from OpenRouter pricing object."""
        return cls(
            model_id=model_id,
            prompt=float(pricing.get("prompt", 0) or 0),
            completion=float(pricing.get("completion", 0) or 0),
            request=float(pricing.get("request", 0) or 0),
            image=float(pricing.get("image", 0) or 0),
            web_search=float(pricing.get("web_search", 0) or 0),
            internal_reasoning=float(pricing.get("internal_reasoning", 0) or 0),
            input_cache_read=float(pricing.get("input_cache_read", 0) or 0),
            input_cache_write=float(pricing.get("input_cache_write", 0) or 0),
        )


@dataclass
class UsageStats:
    """Historical usage statistics for a model/game combination."""
    model_id: str
    game_type: str
    game_count: int = 0
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0

    @property
    def avg_requests_per_game(self) -> float:
        return self.total_requests / self.game_count if self.game_count > 0 else 0

    @property
    def avg_prompt_tokens_per_game(self) -> float:
        return self.total_prompt_tokens / self.game_count if self.game_count > 0 else 0

    @property
    def avg_completion_tokens_per_game(self) -> float:
        return self.total_completion_tokens / self.game_count if self.game_count > 0 else 0

    @property
    def avg_prompt_tokens_per_request(self) -> float:
        return self.total_prompt_tokens / self.total_requests if self.total_requests > 0 else 0

    @property
    def avg_completion_tokens_per_request(self) -> float:
        return self.total_completion_tokens / self.total_requests if self.total_requests > 0 else 0


class CostEstimate(BaseModel):
    """Cost estimate for a game/batch/benchmark."""
    estimated_cost_usd: float
    breakdown: dict[str, float] = {}  # per-model or per-component breakdown
    confidence: str = "medium"  # low, medium, high based on data availability
    notes: list[str] = []  # warnings or explanations


# ============================================
# Pricing Cache
# ============================================

class PricingCache:
    """Cached model pricing from OpenRouter."""

    _instance: "PricingCache | None" = None
    _pricing: dict[str, ModelPricing] = {}
    _last_fetch: float = 0
    _cache_ttl: float = 3600  # 1 hour

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_pricing(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a model, fetching from API if needed."""
        await self._refresh_if_needed()
        return self._pricing.get(model_id)

    async def get_all_pricing(self) -> dict[str, ModelPricing]:
        """Get all cached pricing."""
        await self._refresh_if_needed()
        return self._pricing.copy()

    async def _refresh_if_needed(self):
        """Refresh cache if stale."""
        if time.time() - self._last_fetch > self._cache_ttl:
            await self._fetch_pricing()

    async def _fetch_pricing(self):
        """Fetch pricing from OpenRouter API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get("https://openrouter.ai/api/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])

                    for model in models:
                        model_id = model.get("id", "")
                        pricing = model.get("pricing", {})
                        if model_id and pricing:
                            self._pricing[model_id] = ModelPricing.from_openrouter(
                                model_id, pricing
                            )

                    self._last_fetch = time.time()
                    print(f"[CostEstimator] Fetched pricing for {len(self._pricing)} models")
        except Exception as e:
            print(f"[CostEstimator] Failed to fetch pricing: {e}")
            # Use fallback pricing if fetch fails
            self._load_fallback_pricing()

    def _load_fallback_pricing(self):
        """Load fallback pricing from local config."""
        # Fallback prices per million tokens (convert to per-token)
        fallback = {
            "anthropic/claude-opus-4": {"prompt": 15.0, "completion": 75.0},
            "anthropic/claude-sonnet-4": {"prompt": 3.0, "completion": 15.0},
            "anthropic/claude-haiku-4": {"prompt": 0.25, "completion": 1.25},
            "openai/gpt-4o": {"prompt": 2.5, "completion": 10.0},
            "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
            "openai/o1": {"prompt": 15.0, "completion": 60.0},
            "openai/o1-mini": {"prompt": 1.1, "completion": 4.4},
            "openai/gpt-5.2": {"prompt": 2.0, "completion": 8.0},
            "google/gemini-2.5-pro": {"prompt": 1.25, "completion": 10.0},
            "google/gemini-2.5-flash": {"prompt": 0.15, "completion": 0.6},
            "deepseek/deepseek-r1": {"prompt": 0.55, "completion": 2.19},
            "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.3, "completion": 0.4},
        }

        for model_id, prices in fallback.items():
            self._pricing[model_id] = ModelPricing(
                model_id=model_id,
                prompt=prices["prompt"] / 1_000_000,
                completion=prices["completion"] / 1_000_000,
            )


# ============================================
# Usage Stats Aggregator
# ============================================

def _benchmark_results_dir() -> Path:
    """Get the benchmark_results directory."""
    from src.cloud_benchmark.config import get_data_dir
    return get_data_dir()


def aggregate_usage_stats() -> dict[str, dict[str, UsageStats]]:
    """
    Aggregate usage statistics from all episodes.

    Returns dict mapping model_id -> game_type -> UsageStats.
    """
    stats: dict[str, dict[str, UsageStats]] = defaultdict(
        lambda: defaultdict(lambda: UsageStats(model_id="", game_type=""))
    )

    bench_dir = _benchmark_results_dir()
    if not bench_dir.exists():
        return dict(stats)

    # Scan all episode files
    for json_file in bench_dir.rglob("*.json"):
        if json_file.name.startswith("."):
            continue

        try:
            with open(json_file) as f:
                episode = json.load(f)

            game_type = _detect_game_type(episode, json_file.name)
            if not game_type:
                continue

            # Extract model and usage from episode
            _process_episode_usage(episode, game_type, stats)

        except (json.JSONDecodeError, IOError):
            continue

    return dict(stats)


def _detect_game_type(episode: dict, filename: str) -> str | None:
    """Detect game type from episode."""
    fname = filename.lower()
    if "hanabi" in fname:
        return "hanabi"
    if "decrypto" in fname:
        return "decrypto"
    if "codenames" in fname or ("board" in episode and "winner" in episode):
        return "codenames"
    if "final_score" in episode and "turns" in episode:
        return "hanabi"
    if "code_sequences" in episode or "result_reason" in episode:
        return "decrypto"
    return None


def _process_episode_usage(
    episode: dict,
    game_type: str,
    stats: dict[str, dict[str, UsageStats]]
):
    """Process a single episode and update stats."""
    # Try to get model from metadata
    metadata = episode.get("metadata", {}) or {}
    model = metadata.get("model")

    # For team games, try to get from team config
    if not model:
        for team in ["red_team", "blue_team"]:
            team_data = metadata.get(team, {})
            model = team_data.get("cluer_model") or team_data.get("cluer")
            if model:
                break

    if not model:
        return

    # Initialize stats if needed
    if stats[model][game_type].model_id == "":
        stats[model][game_type].model_id = model
        stats[model][game_type].game_type = game_type

    s = stats[model][game_type]
    s.game_count += 1

    # Aggregate from turn_traces if available
    turn_traces = episode.get("turn_traces", [])
    for turn in turn_traces:
        agent_traces = turn.get("agent_traces", [])
        for trace in agent_traces:
            s.total_requests += 1
            s.total_prompt_tokens += trace.get("input_tokens", 0)
            s.total_completion_tokens += trace.get("output_tokens", 0)

    # Also check turns for Hanabi
    turns = episode.get("turns", [])
    for turn in turns:
        if "input_tokens" in turn or "output_tokens" in turn:
            s.total_requests += 1
            s.total_prompt_tokens += turn.get("input_tokens", 0)
            s.total_completion_tokens += turn.get("output_tokens", 0)

    # Check rounds for Decrypto
    rounds = episode.get("rounds", [])
    for round_data in rounds:
        if isinstance(round_data, dict):
            for key in ["clue_trace", "decode_trace", "intercept_trace"]:
                trace = round_data.get(key, {})
                if trace and isinstance(trace, dict):
                    s.total_requests += 1
                    s.total_prompt_tokens += trace.get("input_tokens", 0)
                    s.total_completion_tokens += trace.get("output_tokens", 0)


# ============================================
# Cost Estimation Functions
# ============================================

async def estimate_game_cost(
    model_id: str,
    game_type: str,
    usage_stats: dict[str, dict[str, UsageStats]] | None = None,
) -> CostEstimate:
    """
    Estimate cost for a single game.

    Args:
        model_id: The model to use
        game_type: codenames, decrypto, or hanabi
        usage_stats: Pre-computed usage stats (optional)

    Returns:
        CostEstimate with estimated cost and breakdown
    """
    cache = PricingCache()
    pricing = await cache.get_pricing(model_id)

    if not pricing:
        # Try without provider prefix
        short_id = model_id.split("/")[-1] if "/" in model_id else model_id
        all_pricing = await cache.get_all_pricing()
        for pid, p in all_pricing.items():
            if short_id in pid:
                pricing = p
                break

    if not pricing:
        return CostEstimate(
            estimated_cost_usd=0.0,
            confidence="low",
            notes=[f"No pricing data available for {model_id}"],
        )

    # Get usage stats
    if usage_stats is None:
        usage_stats = aggregate_usage_stats()

    model_stats = usage_stats.get(model_id, {}).get(game_type)

    # Fallback to global averages for game type
    if not model_stats or model_stats.game_count == 0:
        model_stats = _get_fallback_stats(game_type, usage_stats)
        confidence = "low"
        notes = [f"Using fallback estimates for {model_id} (no historical data)"]
    else:
        confidence = "high" if model_stats.game_count >= 5 else "medium"
        notes = [f"Based on {model_stats.game_count} historical games"]

    # Calculate cost
    prompt_cost = pricing.prompt * model_stats.avg_prompt_tokens_per_game
    completion_cost = pricing.completion * model_stats.avg_completion_tokens_per_game
    request_cost = pricing.request * model_stats.avg_requests_per_game

    total_cost = prompt_cost + completion_cost + request_cost

    return CostEstimate(
        estimated_cost_usd=round(total_cost, 4),
        breakdown={
            "prompt_tokens": round(prompt_cost, 4),
            "completion_tokens": round(completion_cost, 4),
            "requests": round(request_cost, 4),
        },
        confidence=confidence,
        notes=notes,
    )


def _get_fallback_stats(
    game_type: str,
    usage_stats: dict[str, dict[str, UsageStats]]
) -> UsageStats:
    """Get fallback stats by averaging across all models for a game type."""
    fallback = UsageStats(model_id="fallback", game_type=game_type)

    # Aggregate across all models
    total_games = 0
    for model_stats in usage_stats.values():
        if game_type in model_stats:
            s = model_stats[game_type]
            fallback.total_requests += s.total_requests
            fallback.total_prompt_tokens += s.total_prompt_tokens
            fallback.total_completion_tokens += s.total_completion_tokens
            total_games += s.game_count

    fallback.game_count = total_games if total_games > 0 else 1

    # If still no data, use hardcoded defaults
    if fallback.total_requests == 0:
        defaults = {
            "codenames": {"requests": 15, "prompt": 12000, "completion": 800},
            "decrypto": {"requests": 24, "prompt": 18000, "completion": 1200},
            "hanabi": {"requests": 60, "prompt": 45000, "completion": 3000},
        }
        d = defaults.get(game_type, {"requests": 20, "prompt": 15000, "completion": 1000})
        fallback.total_requests = d["requests"]
        fallback.total_prompt_tokens = d["prompt"]
        fallback.total_completion_tokens = d["completion"]
        fallback.game_count = 1

    return fallback


async def estimate_batch_cost(
    games: list[dict],  # List of {model_id, game_type, count}
) -> CostEstimate:
    """
    Estimate cost for a batch of games.

    Args:
        games: List of game configs with model_id, game_type, and count

    Returns:
        CostEstimate with total cost and per-model breakdown
    """
    usage_stats = aggregate_usage_stats()

    total_cost = 0.0
    breakdown = {}
    notes = []
    lowest_confidence = "high"

    for game in games:
        model_id = game.get("model_id", "")
        game_type = game.get("game_type", "")
        count = game.get("count", 1)

        estimate = await estimate_game_cost(model_id, game_type, usage_stats)
        game_cost = estimate.estimated_cost_usd * count

        total_cost += game_cost
        breakdown[f"{model_id}/{game_type}"] = round(game_cost, 4)

        # Track lowest confidence
        if estimate.confidence == "low":
            lowest_confidence = "low"
        elif estimate.confidence == "medium" and lowest_confidence == "high":
            lowest_confidence = "medium"

        notes.extend(estimate.notes)

    return CostEstimate(
        estimated_cost_usd=round(total_cost, 4),
        breakdown=breakdown,
        confidence=lowest_confidence,
        notes=notes[:5],  # Limit notes
    )


async def estimate_benchmark_cost(
    models: list[str],
    game_types: list[str],
    games_per_model_per_type: int,
) -> CostEstimate:
    """
    Estimate cost for a full benchmark run.

    Args:
        models: List of model IDs
        game_types: List of game types to run
        games_per_model_per_type: Number of games per model per game type

    Returns:
        CostEstimate with total cost and breakdown
    """
    games = []
    for model_id in models:
        for game_type in game_types:
            games.append({
                "model_id": model_id,
                "game_type": game_type,
                "count": games_per_model_per_type,
            })

    estimate = await estimate_batch_cost(games)

    # Add summary note
    total_games = len(models) * len(game_types) * games_per_model_per_type
    estimate.notes.insert(0, f"Estimated for {total_games} total games across {len(models)} models")

    return estimate


# ============================================
# API Response Models
# ============================================

class CostEstimateResponse(BaseModel):
    """API response for cost estimation."""
    estimated_cost_usd: float
    estimated_cost_display: str  # Formatted for display
    breakdown: dict[str, float]
    confidence: str
    notes: list[str]

    @classmethod
    def from_estimate(cls, estimate: CostEstimate) -> "CostEstimateResponse":
        cost = estimate.estimated_cost_usd
        if cost < 0.01:
            display = f"${cost:.4f}"
        elif cost < 1.0:
            display = f"${cost:.2f}"
        else:
            display = f"${cost:.2f}"

        return cls(
            estimated_cost_usd=cost,
            estimated_cost_display=display,
            breakdown=estimate.breakdown,
            confidence=estimate.confidence,
            notes=estimate.notes,
        )
