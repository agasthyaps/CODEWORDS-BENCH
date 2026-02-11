"""Tests for model resolution in CloudBenchmarkRunner."""

from __future__ import annotations

import pytest

from src.benchmark.config import ModelConfig
from src.cloud_benchmark.config import CloudBenchmarkConfig
from src.cloud_benchmark.runner import CloudBenchmarkRunner
from src.cloud_benchmark import runner as runner_mod


def test_cloud_runner_resolves_unknown_openrouter_models(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path / "bench"))

    curated = [
        ModelConfig(
            name="curated-sonnet",
            model_id="anthropic/claude-sonnet-4.5",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
    ]
    monkeypatch.setattr(runner_mod, "load_model_farm", lambda _path: (curated, None))

    config = CloudBenchmarkConfig(
        experiment_name="unknown_model_resolution",
        model_ids=["anthropic/claude-sonnet-4.5", "openai/gpt-4o"],
        seed_count=1,
        run_codenames=False,
        run_decrypto=False,
        run_hanabi=True,
        output_dir=str(tmp_path / "out"),
    )

    runner = CloudBenchmarkRunner(config)

    selected_ids = [model.model_id for model in runner._selected_models]
    assert selected_ids == ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"]

    unknown = next(model for model in runner._selected_models if model.model_id == "openai/gpt-4o")
    assert unknown.provider == "openrouter"
    assert unknown.base_url == "https://openrouter.ai/api/v1"
    assert runner.state.hanabi.total_games == 2


def test_cloud_runner_accepts_two_unknown_models_for_competitive_modes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_DATA_DIR", str(tmp_path / "bench"))

    monkeypatch.setattr(runner_mod, "load_model_farm", lambda _path: ([], None))

    config = CloudBenchmarkConfig(
        experiment_name="unknown_models_competitive",
        model_ids=["openai/gpt-4o", "anthropic/claude-sonnet-4.5"],
        seed_count=1,
        run_codenames=True,
        run_decrypto=False,
        run_hanabi=False,
        output_dir=str(tmp_path / "out"),
    )

    runner = CloudBenchmarkRunner(config)
    assert len(runner._selected_models) == 2
    assert runner.state.codenames.total_games > 0
