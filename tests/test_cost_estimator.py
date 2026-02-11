"""Tests for OpenRouter-backed cost estimation."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from src.ui_api import cost_estimator as ce


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    ce.PricingCache._instance = None
    ce.PricingCache._pricing = {}
    ce.PricingCache._last_fetch = 0


def test_aggregate_usage_stats_codenames_turn_trace_shape(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Current codenames turn_traces should produce non-zero model usage."""
    episode = {
        "episode_id": "ep-1",
        "board": {"words": []},
        "winner": "RED",
        "turn_traces": [
            {
                "turn_number": 1,
                "clue_trace": {
                    "agent_id": "red_cluer",
                    "model": "model/a",
                    "input_tokens": 100,
                    "output_tokens": 20,
                },
                "discussion_traces": [
                    {
                        "agent_id": "red_guesser_1",
                        "model": "model/b",
                        "input_tokens": 50,
                        "output_tokens": 10,
                    },
                    {
                        "agent_id": "red_guesser_2",
                        "model": "model/a",
                        "input_tokens": 60,
                        "output_tokens": 12,
                    },
                ],
                "guess_trace": {
                    "agent_id": "red_guesser_1",
                    "model": "model/a",
                    "input_tokens": 30,
                    "output_tokens": 8,
                },
            },
            {
                "turn_number": 2,
                "clue_trace": {
                    "agent_id": "blue_cluer",
                    "model": "model/b",
                    "input_tokens": 70,
                    "output_tokens": 11,
                },
                "discussion_traces": [
                    {
                        "agent_id": "blue_guesser_1",
                        "model": "model/b",
                        "input_tokens": 40,
                        "output_tokens": 6,
                    }
                ],
                "guess_trace": {
                    "agent_id": "blue_guesser_1",
                    "model": "model/b",
                    "input_tokens": 20,
                    "output_tokens": 4,
                },
            },
        ],
    }

    episodes_dir = tmp_path / "exp" / "episodes"
    episodes_dir.mkdir(parents=True)
    with open(episodes_dir / "episode_001.json", "w") as f:
        json.dump(episode, f)

    monkeypatch.setattr(ce, "_benchmark_results_dir", lambda: tmp_path)

    stats = ce.aggregate_usage_stats()

    a = stats["model/a"]["codenames"]
    assert a.game_count == 1
    assert a.total_requests == 3
    assert a.total_prompt_tokens == 190
    assert a.total_completion_tokens == 40

    b = stats["model/b"]["codenames"]
    assert b.game_count == 1
    assert b.total_requests == 4
    assert b.total_prompt_tokens == 180
    assert b.total_completion_tokens == 31


@pytest.mark.asyncio
async def test_fetch_pricing_skips_bad_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed pricing rows should not break fetch for valid models."""

    class _FakeCatalog:
        async def get_models(self, *, text_output_only: bool = False):
            assert text_output_only is False
            return [
                {
                    "id": "good/model",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                },
                {
                    "id": "bad/model",
                    "pricing": {"prompt": "NaN??", "completion": "oops"},
                },
                {
                    "id": "ignored/no-pricing",
                    "pricing": None,
                },
            ]

    monkeypatch.setattr(ce, "OpenRouterCatalogCache", _FakeCatalog)

    cache = ce.PricingCache()
    await cache._fetch_pricing()

    assert "good/model" in cache._pricing
    assert cache._pricing["good/model"].prompt == pytest.approx(0.000001)
    assert cache._pricing["good/model"].completion == pytest.approx(0.000002)

    # Bad row still present, but safely parsed to zero-valued fields.
    assert "bad/model" in cache._pricing
    assert cache._pricing["bad/model"].prompt == 0.0
    assert cache._pricing["bad/model"].completion == 0.0


@pytest.mark.asyncio
async def test_fetch_failure_loads_fallback_and_throttles_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch failures should fallback once and avoid retry thrash until TTL expires."""

    class _FailingCatalog:
        async def get_models(self, *, text_output_only: bool = False):
            raise RuntimeError("network down")

    monkeypatch.setattr(ce, "OpenRouterCatalogCache", _FailingCatalog)

    cache = ce.PricingCache()
    await cache._fetch_pricing()

    assert cache._pricing  # fallback loaded
    assert cache._last_fetch > 0

    refresh_spy = AsyncMock()
    monkeypatch.setattr(cache, "_fetch_pricing", refresh_spy)

    # Fresh fallback timestamp should skip immediate refresh attempts.
    await cache._refresh_if_needed()
    refresh_spy.assert_not_awaited()

    # Once stale, refresh should be attempted.
    cache._last_fetch = time.time() - cache._cache_ttl - 1
    await cache._refresh_if_needed()
    refresh_spy.assert_awaited_once()
