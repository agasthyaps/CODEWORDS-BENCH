#!/usr/bin/env python3
"""Run a benchmark experiment with progress reporting."""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

from src.engine import GameMode, Team
from src.benchmark import (
    ExperimentConfig, ModelConfig, TeamComposition,
    BenchmarkRunner, BenchmarkResult,
    build_leaderboard, export_leaderboard_markdown,
    count_total_games,
    load_model_farm,
)
from src.benchmark.config import MatchupConfig, TeamAssignment


# ANSI colors
class Colors:
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_progress(completed: int, total: int, result: BenchmarkResult):
    """Print progress after each game."""
    pct = (completed / total) * 100
    bar_width = 30
    filled = int(bar_width * completed / total)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Winner color
    if result.winner == Team.RED:
        winner_str = f"{Colors.RED}RED{Colors.RESET}"
    elif result.winner == Team.BLUE:
        winner_str = f"{Colors.BLUE}BLUE{Colors.RESET}"
    else:
        winner_str = f"{Colors.GRAY}DRAW{Colors.RESET}"

    if result.error:
        winner_str = f"{Colors.YELLOW}ERROR{Colors.RESET}"

    # Clear line and print progress
    sys.stdout.write(f"\r{Colors.BOLD}[{bar}]{Colors.RESET} {completed}/{total} ({pct:.1f}%) | Last: {winner_str} | Mode: {result.mode.value}")
    sys.stdout.flush()


def print_game_detail(result: BenchmarkResult):
    """Print detailed game result."""
    if result.error:
        print(f"\n{Colors.YELLOW}⚠ Game failed: {result.error}{Colors.RESET}")
        return

    winner_color = Colors.RED if result.winner == Team.RED else Colors.BLUE if result.winner == Team.BLUE else Colors.GRAY
    winner_str = result.winner.value if result.winner else "DRAW"

    print(f"\n{'─' * 60}")
    print(f"Game: {result.matchup_id}")
    print(f"Mode: {result.mode.value} | Seed: {result.seed}")
    print(f"Winner: {winner_color}{winner_str}{Colors.RESET}")

    if result.metrics:
        print(f"Turns: {result.metrics.turns_to_win}")
        print(f"Red Coord: {result.metrics.red_coordination_score:.3f} | Blue Coord: {result.metrics.blue_coordination_score:.3f}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run a Codenames benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple homogeneous benchmark (models play against each other)
  python run_benchmark.py --models anthropic/claude-3.5-sonnet openai/gpt-4o --seeds 10

  # Custom single matchup with different agents per role
  python run_benchmark.py --custom-matchup \\
      --red-cluer anthropic/claude-3.5-sonnet \\
      --red-guesser openai/gpt-4o \\
      --blue-cluer openai/gpt-4o \\
      --blue-guesser anthropic/claude-3.5-sonnet \\
      --seeds 5

  # Mixed cluer composition (cluers model A, guessers model B)
  python run_benchmark.py --models anthropic/claude-3.5-sonnet openai/gpt-4o \\
      --composition mixed_cluer --seeds 5

  # Mini smoke test (2 models, 1 seed)
  python run_benchmark.py --mini
        """
    )

    # Benchmark mode selection
    parser.add_argument("--name", default="benchmark", help="Experiment name")
    parser.add_argument("--custom-matchup", action="store_true",
                        help="Use custom per-agent model flags instead of --models")

    # Standard benchmark mode (multiple models, generates matchups)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test (space-separated). If omitted and --models-config exists, loads from that file.",
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help="Path to model farm JSON (default: config/models.json if present).",
    )
    parser.add_argument("--composition", choices=["homogeneous", "mixed_cluer", "mixed_guesser"],
                        default="homogeneous", help="Team composition type")

    # Custom matchup mode (per-agent control)
    parser.add_argument("--red-cluer", default=None, help="Model for red cluer")
    parser.add_argument("--red-guesser", default=None, help="Model for both red guessers")
    parser.add_argument("--red-guesser-1", default=None, help="Model for red guesser 1 (overrides --red-guesser)")
    parser.add_argument("--red-guesser-2", default=None, help="Model for red guesser 2 (overrides --red-guesser)")
    parser.add_argument("--blue-cluer", default=None, help="Model for blue cluer")
    parser.add_argument("--blue-guesser", default=None, help="Model for both blue guessers")
    parser.add_argument("--blue-guesser-1", default=None, help="Model for blue guesser 1 (overrides --blue-guesser)")
    parser.add_argument("--blue-guesser-2", default=None, help="Model for blue guesser 2 (overrides --blue-guesser)")

    # Game settings
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (games per matchup)")
    parser.add_argument("--mode", choices=["standard", "no_assassin", "single_guesser"], default="standard",
                        help="Game mode")
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run a small smoke-test benchmark (overrides --seeds and truncates model list).",
    )
    parser.add_argument("--mini-seeds", type=int, default=1, help="Seeds when --mini is set")
    parser.add_argument("--mini-models", type=int, default=2, help="Max models when --mini is set")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and total game count, then exit.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output for each game")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        "standard": GameMode.STANDARD,
        "no_assassin": GameMode.NO_ASSASSIN,
        "single_guesser": GameMode.SINGLE_GUESSER,
    }

    # Map composition string to enum
    comp_map = {
        "homogeneous": TeamComposition.HOMOGENEOUS,
        "mixed_cluer": TeamComposition.MIXED_CLUER,
        "mixed_guesser": TeamComposition.MIXED_GUESSER,
    }

    if args.custom_matchup:
        # Custom matchup mode - build a single matchup from per-agent flags
        default_model = "anthropic/claude-3.5-sonnet"

        red_cluer = args.red_cluer or default_model
        red_guesser_base = args.red_guesser or red_cluer
        red_guesser_1 = args.red_guesser_1 or red_guesser_base
        red_guesser_2 = args.red_guesser_2 or red_guesser_1

        blue_cluer = args.blue_cluer or default_model
        blue_guesser_base = args.blue_guesser or blue_cluer
        blue_guesser_1 = args.blue_guesser_1 or blue_guesser_base
        blue_guesser_2 = args.blue_guesser_2 or blue_guesser_1

        # Collect unique models
        all_models = {red_cluer, red_guesser_1, red_guesser_2, blue_cluer, blue_guesser_1, blue_guesser_2}
        model_configs = [
            ModelConfig(name=m.split("/")[-1] if "/" in m else m, model_id=m)
            for m in all_models
        ]

        def get_model_config(model_id: str) -> ModelConfig:
            return next(m for m in model_configs if m.model_id == model_id)

        custom_matchup = MatchupConfig(
            red_team=TeamAssignment(
                cluer=get_model_config(red_cluer),
                guesser_1=get_model_config(red_guesser_1),
                guesser_2=get_model_config(red_guesser_2),
            ),
            blue_team=TeamAssignment(
                cluer=get_model_config(blue_cluer),
                guesser_1=get_model_config(blue_guesser_1),
                guesser_2=get_model_config(blue_guesser_2),
            ),
            composition=TeamComposition.HETEROGENEOUS,
        )

        config = ExperimentConfig(
            name=args.name,
            description=f"Custom matchup benchmark",
            models=model_configs,
            team_compositions=[],  # We'll override matchup generation
            game_modes=[mode_map[args.mode]],
            seeds=list(range(args.seeds)),
            games_per_config=1,
        )

        # Monkey-patch to use our custom matchup
        import src.benchmark.config as benchmark_config
        original_generate = benchmark_config.generate_matchups
        benchmark_config.generate_matchups = lambda cfg: [custom_matchup]

        description = f"Custom: R({red_cluer.split('/')[-1]}+{red_guesser_1.split('/')[-1]}+{red_guesser_2.split('/')[-1]}) vs B({blue_cluer.split('/')[-1]}+{blue_guesser_1.split('/')[-1]}+{blue_guesser_2.split('/')[-1]})"
    else:
        # Standard benchmark mode - generate matchups from model list
        # Resolve model list: CLI --models > --models-config > default fallback
        repo_root = Path(__file__).parent.parent
        default_models_config_path = repo_root / "config" / "models.json"
        models_config_path = Path(args.models_config) if args.models_config else None

        model_configs: list[ModelConfig] = []
        matchup_strategy = "round_robin"
        matchup_subset = None

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

            if load_path is not None:
                model_configs, farm = load_model_farm(load_path)
                matchup_strategy = farm.default_matchups
                matchup_subset = farm.matchups
            else:
                # Backward-compatible fallback
                fallback = ["anthropic/claude-3.5-sonnet", "openai/gpt-4o"]
                for model_id in fallback:
                    name = model_id.split("/")[-1] if "/" in model_id else model_id
                    model_configs.append(ModelConfig(name=name, model_id=model_id))

        if args.mini:
            # Mini run: cap models and seeds to keep runtime small.
            if args.mini_models > 0:
                model_configs = model_configs[:args.mini_models]
            args.seeds = args.mini_seeds

        config = ExperimentConfig(
            name=args.name,
            description=f"Benchmark: {' vs '.join(m.name for m in model_configs)}",
            models=model_configs,
            team_compositions=[comp_map[args.composition]],
            game_modes=[mode_map[args.mode]],
            seeds=list(range(args.seeds)),
            games_per_config=1,
            matchup_strategy=matchup_strategy,  # type: ignore[arg-type]
            matchup_subset=matchup_subset,
        )
        description = f"{' vs '.join(m.name for m in model_configs)} ({args.composition})"

    total_games = count_total_games(config)

    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}CODENAMES BENCHMARK{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"Experiment: {config.name}")
    print(f"Matchup: {description}")
    print(f"Mode: {args.mode}")
    print(f"Seeds: {args.seeds}")
    if args.mini:
        print(f"{Colors.YELLOW}Mini mode: ON{Colors.RESET} (models={len(config.models)}, seeds={len(config.seeds)})")
    print(f"Total games: {total_games}")
    print(f"Output: benchmark_results/{config.name}/")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("Dry run: exiting before starting benchmark.")
        return

    # Set up logging
    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Progress callback
    def on_progress(completed, total, result):
        if args.verbose:
            print_game_detail(result)
        elif not args.quiet:
            print_progress(completed, total, result)

    # Run benchmark
    start_time = datetime.now()
    runner = BenchmarkRunner(config)
    results = await runner.run(callback=on_progress)
    elapsed = datetime.now() - start_time

    # Print newline after progress bar
    if not args.quiet and not args.verbose:
        print()

    # Summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}BENCHMARK COMPLETE{Colors.RESET}")
    print(f"{'=' * 60}")
    print(f"Time elapsed: {elapsed}")
    print(f"Games completed: {len([r for r in results if not r.error])}")
    print(f"Games failed: {len([r for r in results if r.error])}")

    # Build and print leaderboard
    successful_results = [r for r in results if not r.error]
    if successful_results:
        leaderboard = build_leaderboard(successful_results)
        print(f"\n{export_leaderboard_markdown(leaderboard)}")

        # Save leaderboard to file
        output_dir = Path(config.output_dir) / config.name
        leaderboard_path = output_dir / "leaderboard.md"
        with open(leaderboard_path, "w") as f:
            f.write(export_leaderboard_markdown(leaderboard))
        print(f"\nLeaderboard saved to: {leaderboard_path}")
    else:
        print(f"\n{Colors.YELLOW}No successful games to build leaderboard{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
