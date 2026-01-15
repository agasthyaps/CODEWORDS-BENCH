#!/usr/bin/env python3
"""Run a Decrypto benchmark experiment (standalone from Codenames)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.benchmark import load_model_farm
from src.benchmark.config import ModelConfig
from src.decrypto.benchmark_config import DecryptoExperimentConfig
from src.decrypto.benchmark_runner import DecryptoBenchmarkRunner, DecryptoBenchmarkResult


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_progress(completed: int, total: int, result: DecryptoBenchmarkResult) -> None:
    pct = (completed / total) * 100
    bar_width = 30
    filled = int(bar_width * completed / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    if result.error:
        status = f"{Colors.YELLOW}ERROR{Colors.RESET}"
    elif result.winner:
        status = f"{Colors.GREEN}{result.winner}{Colors.RESET}"
    else:
        status = f"{Colors.GRAY}DRAW{Colors.RESET}"
    sys.stdout.write(f"\r{Colors.BOLD}[{bar}]{Colors.RESET} {completed}/{total} ({pct:.1f}%) | Last: {status}")
    sys.stdout.flush()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Decrypto benchmark")
    parser.add_argument("--name", default="decrypto_benchmark")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mini", action="store_true", help="Mini smoke test (2 models, 1 seed)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    # Resolve models
    repo_root = Path(__file__).parent.parent
    default_models_config_path = repo_root / "config" / "models.json"
    models_config_path = Path(args.models_config) if args.models_config else None

    model_configs: list[ModelConfig] = []
    if args.models:
        for model_id in args.models:
            name = model_id.split("/")[-1] if "/" in model_id else model_id
            model_configs.append(ModelConfig(name=name, model_id=model_id))
    else:
        load_path = None
        if models_config_path and models_config_path.exists():
            load_path = models_config_path
        elif default_models_config_path.exists():
            load_path = default_models_config_path
        if load_path is None:
            raise ValueError("No models provided and no config/models.json found.")
        model_configs, _farm = load_model_farm(load_path)

    if args.mini:
        model_configs = model_configs[:2]
        args.seeds = 1

    config = DecryptoExperimentConfig(
        name=args.name,
        models=model_configs,
        seeds=list(range(args.seeds)),
        games_per_config=1,
        temperature=args.temperature,
    )

    total_games = None
    from src.decrypto.benchmark_config import generate_decrypto_matchups

    total_games = len(generate_decrypto_matchups(config.models)) * len(config.seeds) * config.games_per_config

    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}DECRYPTO BENCHMARK{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"Experiment: {config.name}")
    print(f"Models: {', '.join(m.name for m in config.models)}")
    print(f"Seeds: {len(config.seeds)}")
    print(f"Total games: {total_games}")
    print(f"Output: benchmark_results/{config.name}/")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("Dry run: exiting.")
        return

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def on_progress(completed: int, total: int, result: DecryptoBenchmarkResult) -> None:
        if not args.quiet:
            print_progress(completed, total, result)

    start = datetime.now()
    runner = DecryptoBenchmarkRunner(config)
    results = await runner.run(callback=on_progress)
    elapsed = datetime.now() - start

    if not args.quiet:
        print()

    ok = len([r for r in results if not r.error])
    failed = len([r for r in results if r.error])

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}BENCHMARK COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"Time elapsed: {elapsed}")
    print(f"Games completed: {ok}")
    print(f"Games failed: {failed}")


if __name__ == "__main__":
    asyncio.run(main())

