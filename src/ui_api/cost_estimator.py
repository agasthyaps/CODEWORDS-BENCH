"""Cost estimation module for games, batches, and benchmarks.

Fetches model pricing from OpenRouter and computes cost estimates based on
historical usage data from episodes.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.benchmark.model_farm import load_model_farm

from .openrouter_catalog import OpenRouterCatalogCache

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
        def _safe_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return cls(
            model_id=model_id,
            prompt=_safe_float(pricing.get("prompt", 0)),
            completion=_safe_float(pricing.get("completion", 0)),
            request=_safe_float(pricing.get("request", 0)),
            image=_safe_float(pricing.get("image", 0)),
            web_search=_safe_float(pricing.get("web_search", 0)),
            internal_reasoning=_safe_float(pricing.get("internal_reasoning", 0)),
            input_cache_read=_safe_float(pricing.get("input_cache_read", 0)),
            input_cache_write=_safe_float(pricing.get("input_cache_write", 0)),
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
            catalog = OpenRouterCatalogCache()
            models = await catalog.get_models(text_output_only=False)

            pricing_map: dict[str, ModelPricing] = {}
            for model in models:
                model_id = model.get("id")
                pricing = model.get("pricing")
                if not isinstance(model_id, str) or not model_id:
                    continue
                if not isinstance(pricing, dict):
                    continue

                pricing_map[model_id] = ModelPricing.from_openrouter(model_id, pricing)

            self._pricing = pricing_map
            self._last_fetch = time.time()
            print(f"[CostEstimator] Fetched pricing for {len(self._pricing)} models")
        except Exception as e:
            print(f"[CostEstimator] Failed to fetch pricing: {e}")
            # Use fallback pricing if fetch fails
            self._load_fallback_pricing()
            self._last_fetch = time.time()

    def _load_fallback_pricing(self):
        """Load fallback pricing from local config."""
        # Fallback prices per million tokens (convert to per-token)
        fallback = {
            "anthropic/claude-opus-4": {"prompt": 15.0, "completion": 75.0},
            "anthropic/claude-opus-4.5": {"prompt": 15.0, "completion": 75.0},
            "anthropic/claude-sonnet-4": {"prompt": 3.0, "completion": 15.0},
            "anthropic/claude-sonnet-4.5": {"prompt": 3.0, "completion": 15.0},
            "anthropic/claude-haiku-4": {"prompt": 0.25, "completion": 1.25},
            "anthropic/claude-haiku-4.5": {"prompt": 1.0, "completion": 5.0},
            "openai/gpt-4o": {"prompt": 2.5, "completion": 10.0},
            "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
            "openai/o1": {"prompt": 15.0, "completion": 60.0},
            "openai/o1-mini": {"prompt": 1.1, "completion": 4.4},
            "openai/gpt-5.2": {"prompt": 2.0, "completion": 8.0},
            "google/gemini-2.5-pro": {"prompt": 1.25, "completion": 10.0},
            "google/gemini-2.5-flash": {"prompt": 0.15, "completion": 0.6},
            "google/gemini-3-flash-preview": {"prompt": 0.15, "completion": 0.6},
            "deepseek/deepseek-r1": {"prompt": 0.55, "completion": 2.19},
            "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.3, "completion": 0.4},
            "meta-llama/llama-3.1-405b-instruct": {"prompt": 3.0, "completion": 3.0},
            "moonshotai/kimi-k2.5": {"prompt": 1.2, "completion": 3.6},
            "z-ai/glm-4.7": {"prompt": 0.8, "completion": 2.4},
            "openai/gpt-oss-120b": {"prompt": 0.3, "completion": 0.9},
            "qwen/qwen3-235b-a22b-2507": {"prompt": 0.35, "completion": 1.0},
            "minimax/minimax-m2": {"prompt": 0.9, "completion": 2.7},
        }

        try:
            models, _ = load_model_farm("config/models.json")
            for model in models:
                fallback.setdefault(
                    model.model_id,
                    {"prompt": 2.0, "completion": 8.0},
                )
        except Exception:
            pass

        self._pricing = {}
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
    metadata = episode.get("metadata", {}) or {}
    agent_model_map = _extract_agent_model_map(metadata)
    models_seen: set[str] = set()

    _collect_codenames_usage(episode, game_type, stats, agent_model_map, models_seen)
    _collect_hanabi_usage(episode, game_type, stats, agent_model_map, models_seen)
    _collect_legacy_round_usage(episode, game_type, stats, agent_model_map, models_seen)

    # Count each model at most once per episode.
    for model_id in models_seen:
        s = _ensure_usage_stats(stats, model_id, game_type)
        s.game_count += 1


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ensure_usage_stats(
    stats: dict[str, dict[str, UsageStats]],
    model_id: str,
    game_type: str,
) -> UsageStats:
    if stats[model_id][game_type].model_id == "":
        stats[model_id][game_type].model_id = model_id
        stats[model_id][game_type].game_type = game_type
    return stats[model_id][game_type]


def _extract_agent_model_map(metadata: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}

    direct = metadata.get("agent_models")
    if isinstance(direct, dict):
        for agent_id, model_id in direct.items():
            if isinstance(agent_id, str) and isinstance(model_id, str) and model_id:
                out[agent_id] = model_id

    for team_key in ("red_team", "blue_team"):
        team_data = metadata.get(team_key)
        if not isinstance(team_data, dict):
            continue
        team_agent_models = team_data.get("agent_models")
        if isinstance(team_agent_models, dict):
            for agent_id, model_id in team_agent_models.items():
                if isinstance(agent_id, str) and isinstance(model_id, str) and model_id:
                    out[agent_id] = model_id

    return out


def _record_trace_usage(
    trace: Any,
    game_type: str,
    stats: dict[str, dict[str, UsageStats]],
    agent_model_map: dict[str, str],
    models_seen: set[str],
) -> None:
    if not isinstance(trace, dict):
        return

    model_id = trace.get("model")
    if not isinstance(model_id, str) or not model_id:
        agent_id = trace.get("agent_id")
        if isinstance(agent_id, str):
            model_id = agent_model_map.get(agent_id, "")

    if not isinstance(model_id, str) or not model_id:
        return

    s = _ensure_usage_stats(stats, model_id, game_type)
    s.total_requests += 1
    s.total_prompt_tokens += _safe_int(trace.get("input_tokens"))
    s.total_completion_tokens += _safe_int(trace.get("output_tokens"))
    models_seen.add(model_id)


def _collect_codenames_usage(
    episode: dict,
    game_type: str,
    stats: dict[str, dict[str, UsageStats]],
    agent_model_map: dict[str, str],
    models_seen: set[str],
) -> None:
    turn_traces = episode.get("turn_traces", [])
    if not isinstance(turn_traces, list):
        return

    for turn in turn_traces:
        if not isinstance(turn, dict):
            continue

        # Legacy trace layout.
        for trace in turn.get("agent_traces", []) or []:
            _record_trace_usage(trace, game_type, stats, agent_model_map, models_seen)

        # Current trace layout in codenames episodes.
        _record_trace_usage(turn.get("clue_trace"), game_type, stats, agent_model_map, models_seen)
        _record_trace_usage(turn.get("guess_trace"), game_type, stats, agent_model_map, models_seen)

        for trace in turn.get("discussion_traces", []) or []:
            _record_trace_usage(trace, game_type, stats, agent_model_map, models_seen)


def _collect_hanabi_usage(
    episode: dict,
    game_type: str,
    stats: dict[str, dict[str, UsageStats]],
    agent_model_map: dict[str, str],
    models_seen: set[str],
) -> None:
    turns = episode.get("turns", [])
    if not isinstance(turns, list):
        return

    metadata = episode.get("metadata", {}) or {}
    player_models = metadata.get("player_models", {})
    default_model = metadata.get("model")

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if "input_tokens" not in turn and "output_tokens" not in turn:
            continue

        model_id = turn.get("model")
        if not isinstance(model_id, str) or not model_id:
            player_id = turn.get("player_id")
            if isinstance(player_models, dict) and isinstance(player_id, str):
                mapped = player_models.get(player_id)
                if isinstance(mapped, str):
                    model_id = mapped
        if (not isinstance(model_id, str) or not model_id) and isinstance(default_model, str):
            model_id = default_model

        _record_trace_usage(
            {
                "model": model_id,
                "input_tokens": turn.get("input_tokens", 0),
                "output_tokens": turn.get("output_tokens", 0),
                "agent_id": turn.get("player_id"),
            },
            game_type,
            stats,
            agent_model_map,
            models_seen,
        )


def _collect_legacy_round_usage(
    episode: dict,
    game_type: str,
    stats: dict[str, dict[str, UsageStats]],
    agent_model_map: dict[str, str],
    models_seen: set[str],
) -> None:
    rounds = episode.get("rounds", [])
    if not isinstance(rounds, list):
        return

    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        for key in ("clue_trace", "decode_trace", "intercept_trace"):
            _record_trace_usage(round_data.get(key), game_type, stats, agent_model_map, models_seen)


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
    has_model_usage = bool(
        model_stats
        and model_stats.game_count > 0
        and (
            model_stats.total_requests > 0
            or model_stats.total_prompt_tokens > 0
            or model_stats.total_completion_tokens > 0
        )
    )

    # Fallback to global averages for game type
    if not has_model_usage:
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
