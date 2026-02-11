#!/usr/bin/env python3
"""
Overnight benchmark script for Codenames, Decrypto, and Hanabi.

Runs comprehensive multi-game benchmarks with crash recovery.
Designed to run unattended overnight.

CRASH RECOVERY:
    Progress is saved after each game. If the script crashes or is interrupted,
    simply run it again and it will resume from where it left off.

ESTIMATED RUNTIME (30 seeds, 6 models):
    - Codenames: ~3600 games × ~2-3 min/game = 120-180 hours
    - Decrypto:  ~1800 games × ~3-5 min/game = 90-150 hours  
    - Hanabi:    ~180 games × ~5-10 min/game = 15-30 hours
    
    For overnight (~8 hours), use --seeds 3 for a smaller run.

Usage:
    # Full overnight run (WARNING: takes many hours!)
    python scripts/run_overnight_benchmark.py
    
    # Smaller test run (recommended for initial testing)
    python scripts/run_overnight_benchmark.py --seeds 3
    
    # Dry run to see config without running
    python scripts/run_overnight_benchmark.py --dry-run
    
    # Skip specific games
    python scripts/run_overnight_benchmark.py --skip-hanabi --seeds 5
    
    # Run in background with nohup
    nohup python scripts/run_overnight_benchmark.py --seeds 10 > benchmark.log 2>&1 &
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment before importing src modules
load_dotenv(Path(__file__).parent.parent / ".env")

from src.agents.llm import create_provider
from src.benchmark import (
    ExperimentConfig,
    ModelConfig,
    TeamComposition,
    BenchmarkRunner,
    BenchmarkResult,
    build_leaderboard,
    export_leaderboard_markdown,
    count_total_games,
)
from src.benchmark.config import MatchupConfig, TeamAssignment
from src.core.state import AgentStateManager
from src.decrypto.benchmark_config import DecryptoExperimentConfig, generate_decrypto_matchups
from src.decrypto.benchmark_runner import DecryptoBenchmarkRunner, DecryptoBenchmarkResult
from src.engine import GameMode, Team
from src.hanabi.agents.llm_agent import HanabiPlayerLLM
from src.hanabi.models import HanabiConfig, HanabiEpisodeRecord
from src.hanabi.orchestrator import run_episode as run_hanabi_episode
from src.hanabi.metrics import compute_episode_metrics as compute_hanabi_metrics

# =============================================================================
# Configuration
# =============================================================================

# Models to benchmark (swap sonnet-4.5 for gemini-3-flash, add glm-4.7)
BENCHMARK_MODELS = [
    ModelConfig(name="claude-4.5-opus", model_id="anthropic/claude-opus-4.5"),
    ModelConfig(name="gpt-4o", model_id="openai/gpt-4o"),
    ModelConfig(name="gpt-5.2", model_id="openai/gpt-5.2"),
    ModelConfig(name="claude-haiku-4.5", model_id="anthropic/claude-haiku-4.5"),
    ModelConfig(name="gemini-3-flash", model_id="google/gemini-3-flash-preview"),
    ModelConfig(name="glm-4.7", model_id="z-ai/glm-4.7"),
]

# Experiment settings (defaults, can be overridden via CLI)
DEFAULT_NUM_SEEDS = 30  # For statistical power
TEMPERATURE = 0.7
MAX_DISCUSSION_ROUNDS = 3
MAX_TURNS = 50

# Output directory
OUTPUT_DIR = Path("benchmark_results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"overnight_{TIMESTAMP}"

# Runtime config (set in main)
num_seeds = DEFAULT_NUM_SEEDS


# =============================================================================
# Console formatting
# =============================================================================

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}▶ {title}{Colors.RESET}")
    print(f"{Colors.GRAY}{'-' * 60}{Colors.RESET}")


def print_progress(completed: int, total: int, status: str) -> None:
    pct = (completed / total) * 100 if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * completed / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r{Colors.BOLD}[{bar}]{Colors.RESET} {completed}/{total} ({pct:.1f}%) | {status}")
    sys.stdout.flush()


# =============================================================================
# Hanabi Benchmark Runner (not in src, so we implement here)
# =============================================================================

class HanabiBenchmarkProgress:
    """Track Hanabi benchmark progress for crash recovery."""
    
    def __init__(self, path: Path, total_games: int):
        self.path = path
        self.total_games = total_games
        self.completed_keys: set[str] = set()
        self.completed_games = 0
        self.failed_games = 0
        self._load()
    
    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "r") as f:
                data = json.load(f)
            self.completed_keys = set(data.get("completed_keys", []))
            self.completed_games = data.get("completed_games", 0)
            self.failed_games = data.get("failed_games", 0)
    
    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({
                "total_games": self.total_games,
                "completed_keys": list(self.completed_keys),
                "completed_games": self.completed_games,
                "failed_games": self.failed_games,
            }, f, indent=2)
    
    def key(self, model_combo: str, seed: int) -> str:
        return f"{model_combo}|{seed}"
    
    def is_completed(self, model_combo: str, seed: int) -> bool:
        return self.key(model_combo, seed) in self.completed_keys
    
    def mark_completed(self, model_combo: str, seed: int) -> None:
        self.completed_keys.add(self.key(model_combo, seed))
        self.completed_games += 1
        self.save()
    
    def mark_failed(self) -> None:
        self.failed_games += 1
        self.save()


async def run_hanabi_benchmark(
    models: list[ModelConfig],
    seeds: list[int],
    output_dir: Path,
    callback: callable = None,
) -> list[dict[str, Any]]:
    """
    Run Hanabi benchmark across model combinations.
    
    For Hanabi (cooperative), we test teams of 3 players from the same model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    progress_path = output_dir / "progress.json"
    
    # Calculate total games: each model × each seed
    total_games = len(models) * len(seeds)
    progress = HanabiBenchmarkProgress(progress_path, total_games)
    
    results: list[dict[str, Any]] = []
    
    for model in models:
        model_combo = model.model_id
        
        for seed in seeds:
            if progress.is_completed(model_combo, seed):
                continue
            
            try:
                # Create 3 players with the same model
                provider = create_provider("openrouter", model.model_id)
                players = [
                    HanabiPlayerLLM(provider=provider, player_id=f"player_{i}", temperature=TEMPERATURE)
                    for i in range(3)
                ]
                
                config = HanabiConfig(num_players=3, hand_size=5, seed=seed)
                agent_states = AgentStateManager()
                
                start_time = time.time()
                episode = await run_hanabi_episode(
                    config=config,
                    players=players,
                    agent_states=agent_states,
                    episode_id=f"hanabi_{model.name}_{seed:04d}",
                    metadata={"model": model.model_id, "seed": seed},
                )
                duration = time.time() - start_time
                
                # Save episode
                episode.save(str(episodes_dir))
                
                # Compute metrics
                metrics = compute_hanabi_metrics(episode)
                
                result = {
                    "model": model.model_id,
                    "model_name": model.name,
                    "seed": seed,
                    "episode_id": episode.episode_id,
                    "score": episode.final_score,
                    "game_over_reason": episode.game_over_reason,
                    "duration_seconds": duration,
                    "metrics": metrics,
                    "error": None,
                }
                
                progress.mark_completed(model_combo, seed)
                
            except Exception as e:
                result = {
                    "model": model.model_id,
                    "model_name": model.name,
                    "seed": seed,
                    "episode_id": "FAILED",
                    "score": 0,
                    "game_over_reason": "error",
                    "duration_seconds": 0,
                    "metrics": {},
                    "error": str(e),
                }
                progress.mark_failed()
            
            results.append(result)
            
            # Append result to file
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            
            if callback:
                callback(progress.completed_games, total_games, result)
    
    return results


def export_hanabi_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Export Hanabi benchmark summary to markdown."""

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for r in results:
        if r["error"]:
            continue
        model = r["model_name"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    lines = ["# Hanabi Benchmark Results", ""]

    model_rows: list[dict[str, Any]] = []
    all_efficiencies: list[float] = []

    for model, games in by_model.items():
        n = len(games)
        if n == 0:
            continue
        scores = [int(g.get("score", 0)) for g in games]
        turns = [
            int((g.get("metrics") or {}).get("total_turns", 0) or 0)
            for g in games
        ]
        efficiencies = [
            (score / max(turn_count, 1))
            for score, turn_count in zip(scores, turns)
        ]
        all_efficiencies.extend(efficiencies)

        avg_score = sum(scores) / n
        max_score = max(scores)
        avg_pct = avg_score / 25 * 100
        perfect = sum(1 for s in scores if s == 25)
        avg_turns = sum(turns) / n if turns else 0.0
        avg_efficiency = sum(efficiencies) / n if efficiencies else 0.0
        turn_limit_hits = sum(
            1
            for g, t in zip(games, turns)
            if t >= 200 or g.get("game_over_reason") == "turn_limit"
        )
        turn_limit_pct = (turn_limit_hits / n) * 100 if n > 0 else 0.0

        model_rows.append(
            {
                "model": model,
                "games": n,
                "avg_efficiency": avg_efficiency,
                "avg_turns": avg_turns,
                "avg_score": avg_score,
                "max_score": max_score,
                "avg_pct": avg_pct,
                "perfect": perfect,
                "turn_limit_pct": turn_limit_pct,
            }
        )

    model_rows.sort(
        key=lambda row: (row["avg_efficiency"], row["avg_score"], row["avg_pct"]),
        reverse=True,
    )

    # Efficiency-first summary table
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| Rank | Model | Games | Avg Efficiency | Avg Turns | Avg Score | Avg Score % | Turn-Limit % | Perfect (25) |")
    lines.append("|------|-------|-------|----------------|-----------|-----------|-------------|--------------|--------------|")

    for rank, row in enumerate(model_rows, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['games']} | "
            f"{row['avg_efficiency']:.4f} | {row['avg_turns']:.1f} | "
            f"{row['avg_score']:.1f} | {row['avg_pct']:.1f}% | "
            f"{row['turn_limit_pct']:.1f}% | {row['perfect']} |"
        )

    lines.append("")

    overall_efficiency = (
        sum(all_efficiencies) / len(all_efficiencies)
        if all_efficiencies
        else None
    )
    tom_block = {
        "cluer_calibration_brier": {
            "value": None,
            "n": 0,
            "ci_method": "none",
            "exclusions": ["requires_codenames_cluer_predictions"],
            "parse_failure_rate": None,
        },
        "cluer_bias": {
            "value": None,
            "n": 0,
            "ci_method": "none",
            "exclusions": ["requires_codenames_cluer_predictions"],
            "parse_failure_rate": None,
        },
        "alignment_f1": {
            "value": None,
            "n": 0,
            "ci_method": "none",
            "exclusions": ["requires_codenames_alignment_traces"],
            "parse_failure_rate": None,
        },
        "intercept_gap": {
            "value": None,
            "n": 0,
            "ci_method": "none",
            "exclusions": ["requires_decrypto_intercept_metrics"],
            "parse_failure_rate": None,
        },
        "hanabi_efficiency": {
            "value": round(overall_efficiency, 6) if overall_efficiency is not None else None,
            "n": len(all_efficiencies),
            "ci_method": "mean_no_ci",
            "exclusions": [],
            "parse_failure_rate": None,
        },
    }

    lines.append("## ToM Block")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(tom_block, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Robustness Block")
    lines.append("")
    lines.append("```json")
    lines.append("[]")
    lines.append("```")
    lines.append("")

    # Save
    with open(output_dir / "hanabi_summary.md", "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main Benchmark Runner
# =============================================================================

async def run_codenames_benchmark(dry_run: bool = False) -> list[BenchmarkResult]:
    """Run Codenames benchmark."""
    
    print_section("Codenames Benchmark")
    
    config = ExperimentConfig(
        name=f"{EXPERIMENT_NAME}_codenames",
        description="Overnight Codenames benchmark",
        models=BENCHMARK_MODELS,
        team_compositions=[TeamComposition.HOMOGENEOUS, TeamComposition.MIXED_CLUER],
        game_modes=[GameMode.STANDARD],
        seeds=list(range(num_seeds)),
        games_per_config=1,
        temperature=TEMPERATURE,
        max_discussion_rounds=MAX_DISCUSSION_ROUNDS,
        max_turns=MAX_TURNS,
    )
    
    total_games = count_total_games(config)
    print(f"  Models: {', '.join(m.name for m in BENCHMARK_MODELS)}")
    print(f"  Compositions: homogeneous, mixed_cluer")
    print(f"  Seeds: {num_seeds}")
    print(f"  Total games: {total_games}")
    
    if dry_run:
        print(f"  {Colors.YELLOW}[DRY RUN - skipping]{Colors.RESET}")
        return []
    
    def on_progress(completed: int, total: int, result: BenchmarkResult):
        winner = result.winner.value if result.winner else "DRAW"
        if result.error:
            status = f"{Colors.YELLOW}ERROR{Colors.RESET}"
        else:
            status = f"{Colors.GREEN}{winner}{Colors.RESET}"
        print_progress(completed, total, f"Last: {status}")
    
    runner = BenchmarkRunner(config)
    results = await runner.run(callback=on_progress)
    print()  # Newline after progress bar
    
    # Build and save leaderboard
    successful = [r for r in results if not r.error]
    if successful:
        leaderboard = build_leaderboard(successful)
        leaderboard_md = export_leaderboard_markdown(leaderboard)
        output_dir = OUTPUT_DIR / config.name
        with open(output_dir / "leaderboard.md", "w") as f:
            f.write(leaderboard_md)
        print(f"  {Colors.GREEN}✓ Saved leaderboard to {output_dir / 'leaderboard.md'}{Colors.RESET}")
    
    return results


async def run_decrypto_benchmark_task(dry_run: bool = False) -> list[DecryptoBenchmarkResult]:
    """Run Decrypto benchmark."""
    
    print_section("Decrypto Benchmark")
    
    config = DecryptoExperimentConfig(
        name=f"{EXPERIMENT_NAME}_decrypto",
        models=BENCHMARK_MODELS,
        seeds=list(range(num_seeds)),
        games_per_config=1,
        temperature=TEMPERATURE,
    )
    
    matchups = generate_decrypto_matchups(config.models)
    total_games = len(matchups) * len(config.seeds) * config.games_per_config
    
    print(f"  Models: {', '.join(m.name for m in BENCHMARK_MODELS)}")
    print(f"  Matchups: {len(matchups)}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Total games: {total_games}")
    
    if dry_run:
        print(f"  {Colors.YELLOW}[DRY RUN - skipping]{Colors.RESET}")
        return []
    
    def on_progress(completed: int, total: int, result: DecryptoBenchmarkResult):
        if result.error:
            status = f"{Colors.YELLOW}ERROR{Colors.RESET}"
        elif result.winner:
            status = f"{Colors.GREEN}{result.winner}{Colors.RESET}"
        else:
            status = f"{Colors.GRAY}DRAW{Colors.RESET}"
        print_progress(completed, total, f"Last: {status}")
    
    runner = DecryptoBenchmarkRunner(config)
    results = await runner.run(callback=on_progress)
    print()  # Newline after progress bar
    
    return results


async def run_hanabi_benchmark_task(dry_run: bool = False) -> list[dict[str, Any]]:
    """Run Hanabi benchmark."""
    
    print_section("Hanabi Benchmark")
    
    output_dir = OUTPUT_DIR / f"{EXPERIMENT_NAME}_hanabi"
    total_games = len(BENCHMARK_MODELS) * num_seeds
    
    print(f"  Models: {', '.join(m.name for m in BENCHMARK_MODELS)}")
    print(f"  Seeds: {num_seeds}")
    print(f"  Total games: {total_games}")
    
    if dry_run:
        print(f"  {Colors.YELLOW}[DRY RUN - skipping]{Colors.RESET}")
        return []
    
    def on_progress(completed: int, total: int, result: dict):
        if result.get("error"):
            status = f"{Colors.YELLOW}ERROR{Colors.RESET}"
        else:
            score = result.get("score", 0)
            status = f"{Colors.GREEN}Score: {score}/25{Colors.RESET}"
        print_progress(completed, total, f"Last: {status}")
    
    results = await run_hanabi_benchmark(
        models=BENCHMARK_MODELS,
        seeds=list(range(num_seeds)),
        output_dir=output_dir,
        callback=on_progress,
    )
    print()  # Newline after progress bar
    
    # Export summary
    export_hanabi_summary(results, output_dir)
    print(f"  {Colors.GREEN}✓ Saved summary to {output_dir / 'hanabi_summary.md'}{Colors.RESET}")
    
    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Run overnight benchmarks for Codenames, Decrypto, and Hanabi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without running")
    parser.add_argument("--skip-codenames", action="store_true", help="Skip Codenames benchmark")
    parser.add_argument("--skip-decrypto", action="store_true", help="Skip Decrypto benchmark")
    parser.add_argument("--skip-hanabi", action="store_true", help="Skip Hanabi benchmark")
    parser.add_argument("--seeds", type=int, default=DEFAULT_NUM_SEEDS, help=f"Number of seeds (default: {DEFAULT_NUM_SEEDS})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    # Update runtime seeds config
    global num_seeds
    num_seeds = args.seeds
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)
    
    print_header("OVERNIGHT BENCHMARK SUITE")
    
    print(f"  {Colors.BOLD}Experiment:{Colors.RESET} {EXPERIMENT_NAME}")
    print(f"  {Colors.BOLD}Output:{Colors.RESET} {OUTPUT_DIR}")
    print(f"  {Colors.BOLD}Models:{Colors.RESET} {', '.join(m.name for m in BENCHMARK_MODELS)}")
    print(f"  {Colors.BOLD}Seeds:{Colors.RESET} {num_seeds}")
    print(f"  {Colors.BOLD}Temperature:{Colors.RESET} {TEMPERATURE}")
    
    if args.dry_run:
        print(f"\n  {Colors.YELLOW}*** DRY RUN MODE ***{Colors.RESET}\n")
    
    start_time = datetime.now()
    
    # Track results for summary
    codenames_results = []
    decrypto_results = []
    hanabi_results = []
    
    try:
        # Codenames
        if not args.skip_codenames:
            codenames_results = await run_codenames_benchmark(dry_run=args.dry_run)
        else:
            print_section("Codenames Benchmark")
            print(f"  {Colors.GRAY}[SKIPPED]{Colors.RESET}")
        
        # Decrypto
        if not args.skip_decrypto:
            decrypto_results = await run_decrypto_benchmark_task(dry_run=args.dry_run)
        else:
            print_section("Decrypto Benchmark")
            print(f"  {Colors.GRAY}[SKIPPED]{Colors.RESET}")
        
        # Hanabi
        if not args.skip_hanabi:
            hanabi_results = await run_hanabi_benchmark_task(dry_run=args.dry_run)
        else:
            print_section("Hanabi Benchmark")
            print(f"  {Colors.GRAY}[SKIPPED]{Colors.RESET}")
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Progress has been saved.{Colors.RESET}")
        print(f"Run the script again to resume from where it left off.")
    
    except Exception as e:
        print(f"\n\n{Colors.RED}Error: {e}{Colors.RESET}")
        print(f"Progress has been saved. Run the script again to resume.")
        logging.exception("Benchmark failed")
        raise
    
    # Final summary
    elapsed = datetime.now() - start_time
    
    print_header("BENCHMARK COMPLETE")
    print(f"  {Colors.BOLD}Time elapsed:{Colors.RESET} {elapsed}")
    
    if codenames_results:
        ok = len([r for r in codenames_results if not r.error])
        fail = len([r for r in codenames_results if r.error])
        print(f"  {Colors.BOLD}Codenames:{Colors.RESET} {ok} completed, {fail} failed")
    
    if decrypto_results:
        ok = len([r for r in decrypto_results if not r.error])
        fail = len([r for r in decrypto_results if r.error])
        print(f"  {Colors.BOLD}Decrypto:{Colors.RESET} {ok} completed, {fail} failed")
    
    if hanabi_results:
        ok = len([r for r in hanabi_results if not r.get("error")])
        fail = len([r for r in hanabi_results if r.get("error")])
        print(f"  {Colors.BOLD}Hanabi:{Colors.RESET} {ok} completed, {fail} failed")
    
    print(f"\n  Results saved to: {Colors.CYAN}{OUTPUT_DIR}{Colors.RESET}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
