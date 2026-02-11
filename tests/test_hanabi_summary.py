from __future__ import annotations

from pathlib import Path

from scripts.run_overnight_benchmark import export_hanabi_summary


def test_hanabi_summary_is_efficiency_first(tmp_path: Path) -> None:
    results = [
        {
            "model": "openai/model-a",
            "model_name": "model-a",
            "seed": 1,
            "score": 20,
            "game_over_reason": "completed",
            "metrics": {"total_turns": 100},
            "error": None,
        },
        {
            "model": "openai/model-a",
            "model_name": "model-a",
            "seed": 2,
            "score": 20,
            "game_over_reason": "completed",
            "metrics": {"total_turns": 100},
            "error": None,
        },
        {
            "model": "openai/model-b",
            "model_name": "model-b",
            "seed": 1,
            "score": 25,
            "game_over_reason": "completed",
            "metrics": {"total_turns": 190},
            "error": None,
        },
        {
            "model": "openai/model-b",
            "model_name": "model-b",
            "seed": 2,
            "score": 24,
            "game_over_reason": "completed",
            "metrics": {"total_turns": 180},
            "error": None,
        },
    ]

    export_hanabi_summary(results, tmp_path)
    summary = (tmp_path / "hanabi_summary.md").read_text()

    assert "## ToM Block" in summary
    assert "## Robustness Block" in summary
    assert "hanabi_efficiency" in summary
    assert "Turn-Limit %" in summary

    model_a_index = summary.find("| 1 | model-a ")
    model_b_index = summary.find("| 2 | model-b ")
    assert model_a_index != -1
    assert model_b_index != -1
    assert model_a_index < model_b_index
