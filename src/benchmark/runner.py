"""Benchmark runner with crash recovery (M6)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.engine import Team, GameConfig, GameMode
from src.agents import AgentConfig, CluerAgent, GuesserAgent, create_provider
from src.runner import TeamAgents, run_episode, ExtendedEpisodeRecord
from src.metrics import compute_episode_metrics, EpisodeMetrics

from .config import (
    ExperimentConfig, MatchupConfig, TeamAssignment, ModelConfig,
    generate_matchups, count_total_games,
)


logger = logging.getLogger(__name__)


class BenchmarkResult(BaseModel):
    """Result from a single benchmark game."""
    matchup_id: str
    mode: GameMode
    seed: int
    game_index: int
    episode_id: str
    winner: Team | None
    metrics: EpisodeMetrics
    red_models: dict[str, str]  # role -> model_id
    blue_models: dict[str, str]
    duration_seconds: float
    error: str | None = None


class BenchmarkProgress(BaseModel):
    """Progress tracking for crash recovery."""
    experiment_name: str
    started_at: datetime
    total_games: int
    completed_games: int
    failed_games: int
    skipped_configs: list[str] = Field(default_factory=list)
    completed_keys: set[str] = Field(default_factory=set)

    def get_progress_key(
        self,
        matchup_id: str,
        mode: GameMode,
        seed: int,
        game_index: int,
    ) -> str:
        """Generate a unique key for a specific game configuration."""
        return f"{matchup_id}|{mode.value}|{seed}|{game_index}"

    def is_completed(
        self,
        matchup_id: str,
        mode: GameMode,
        seed: int,
        game_index: int,
    ) -> bool:
        """Check if a specific game has been completed."""
        key = self.get_progress_key(matchup_id, mode, seed, game_index)
        return key in self.completed_keys

    def mark_completed(
        self,
        matchup_id: str,
        mode: GameMode,
        seed: int,
        game_index: int,
    ) -> None:
        """Mark a game as completed."""
        key = self.get_progress_key(matchup_id, mode, seed, game_index)
        self.completed_keys.add(key)
        self.completed_games += 1

    def mark_failed(self) -> None:
        """Increment failed game counter."""
        self.failed_games += 1

    model_config = {"arbitrary_types_allowed": True}


def _get_matchup_id(matchup: MatchupConfig) -> str:
    """Generate a unique ID for a matchup."""
    red = f"R({matchup.red_team.cluer.name},{matchup.red_team.guesser_1.name},{matchup.red_team.guesser_2.name})"
    blue = f"B({matchup.blue_team.cluer.name},{matchup.blue_team.guesser_1.name},{matchup.blue_team.guesser_2.name})"

    meta = []
    if matchup.pair_key:
        meta.append(f"pair={matchup.pair_key}")
    if matchup.config_type:
        meta.append(f"cfg={matchup.config_type}")
    if matchup.direction:
        meta.append(f"dir={matchup.direction}")
    meta_str = "|".join(meta) if meta else matchup.composition.value

    return f"{meta_str}:{red}vs{blue}"


def _build_team_agents(
    team_assignment: TeamAssignment,
    team: Team,
    temperature: float,
    max_retries: int,
) -> TeamAgents:
    """Build TeamAgents from a TeamAssignment."""
    team_key = team.value.lower()

    # Create providers
    cluer_provider = create_provider(
        team_assignment.cluer.provider,
        team_assignment.cluer.model_id,
        base_url=team_assignment.cluer.base_url,
    )
    guesser1_provider = create_provider(
        team_assignment.guesser_1.provider,
        team_assignment.guesser_1.model_id,
        base_url=team_assignment.guesser_1.base_url,
    )
    guesser2_provider = create_provider(
        team_assignment.guesser_2.provider,
        team_assignment.guesser_2.model_id,
        base_url=team_assignment.guesser_2.base_url,
    )

    # Create agents
    cluer = CluerAgent(
        AgentConfig(
            model=team_assignment.cluer.model_id,
            role="cluer",
            team=team,
            agent_id=f"{team_key}_cluer",
            temperature=temperature,
            max_retries=max_retries,
        ),
        cluer_provider,
    )

    guesser1 = GuesserAgent(
        AgentConfig(
            model=team_assignment.guesser_1.model_id,
            role="guesser",
            team=team,
            agent_id=f"{team_key}_guesser_1",
            temperature=temperature,
        ),
        guesser1_provider,
    )

    guesser2 = GuesserAgent(
        AgentConfig(
            model=team_assignment.guesser_2.model_id,
            role="guesser",
            team=team,
            agent_id=f"{team_key}_guesser_2",
            temperature=temperature,
        ),
        guesser2_provider,
    )

    return TeamAgents(cluer=cluer, guesser_1=guesser1, guesser_2=guesser2)


async def run_single_game(
    matchup: MatchupConfig,
    mode: GameMode,
    seed: int,
    config: ExperimentConfig,
    emit_fn: Any | None = None,
) -> tuple[ExtendedEpisodeRecord, EpisodeMetrics]:
    """Run a single benchmark game."""
    game_config = GameConfig.for_mode(mode, seed=seed)

    red_team = _build_team_agents(
        matchup.red_team,
        Team.RED,
        config.temperature,
        config.max_retries,
    )
    blue_team = _build_team_agents(
        matchup.blue_team,
        Team.BLUE,
        config.temperature,
        config.max_retries,
    )

    episode = await run_episode(
        config=game_config,
        red_team=red_team,
        blue_team=blue_team,
        max_turns=config.max_turns,
        max_discussion_rounds=config.max_discussion_rounds,
        emit_fn=emit_fn,
    )

    metrics = compute_episode_metrics(episode)

    return episode, metrics


class BenchmarkRunner:
    """Runner for benchmark experiments with crash recovery."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.episodes_dir = self.output_dir / "episodes"
        self.results: list[BenchmarkResult] = []
        self.progress: BenchmarkProgress | None = None

    def _setup_output_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(mode="json"), f, indent=2)

    def _load_progress(self) -> BenchmarkProgress:
        """Load progress from disk or create new."""
        progress_path = self.output_dir / "progress.json"
        if progress_path.exists():
            with open(progress_path, "r") as f:
                data = json.load(f)
            # Convert completed_keys back to set
            data["completed_keys"] = set(data.get("completed_keys", []))
            return BenchmarkProgress.model_validate(data)

        return BenchmarkProgress(
            experiment_name=self.config.name,
            started_at=datetime.utcnow(),
            total_games=count_total_games(self.config),
            completed_games=0,
            failed_games=0,
        )

    def _save_progress(self) -> None:
        """Save progress to disk."""
        if self.progress is None:
            return

        progress_path = self.output_dir / "progress.json"
        data = self.progress.model_dump(mode="json")
        # Convert set to list for JSON
        data["completed_keys"] = list(self.progress.completed_keys)
        with open(progress_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_episode(self, episode: ExtendedEpisodeRecord) -> Path:
        """Save an episode to disk."""
        return episode.save(self.episodes_dir)

    def _save_result(self, result: BenchmarkResult) -> None:
        """Append result to results file."""
        results_path = self.output_dir / "results.jsonl"
        with open(results_path, "a") as f:
            f.write(json.dumps(result.model_dump(mode="json")) + "\n")

    def _load_results(self) -> list[BenchmarkResult]:
        """Load existing results from disk."""
        results_path = self.output_dir / "results.jsonl"
        if not results_path.exists():
            return []

        results = []
        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(BenchmarkResult.model_validate(data))
        return results

    async def _run_with_retry(
        self,
        matchup: MatchupConfig,
        mode: GameMode,
        seed: int,
        game_index: int,
    ) -> BenchmarkResult | None:
        """Run a single game with exponential backoff retry."""
        matchup_id = _get_matchup_id(matchup)
        consecutive_failures = 0
        delay = self.config.retry_delay_base

        while consecutive_failures < self.config.max_consecutive_failures:
            try:
                start_time = time.time()
                episode, metrics = await run_single_game(
                    matchup, mode, seed, self.config
                )
                duration = time.time() - start_time

                # Save episode immediately
                self._save_episode(episode)

                result = BenchmarkResult(
                    matchup_id=matchup_id,
                    mode=mode,
                    seed=seed,
                    game_index=game_index,
                    episode_id=episode.episode_id,
                    winner=episode.winner,
                    metrics=metrics,
                    red_models={
                        "cluer": matchup.red_team.cluer.model_id,
                        "guesser_1": matchup.red_team.guesser_1.model_id,
                        "guesser_2": matchup.red_team.guesser_2.model_id,
                    },
                    blue_models={
                        "cluer": matchup.blue_team.cluer.model_id,
                        "guesser_1": matchup.blue_team.guesser_1.model_id,
                        "guesser_2": matchup.blue_team.guesser_2.model_id,
                    },
                    duration_seconds=duration,
                )

                return result

            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    f"Game failed (attempt {consecutive_failures}): {e}"
                )

                if consecutive_failures < self.config.max_consecutive_failures:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.config.retry_delay_max)

        # Max failures reached
        logger.error(
            f"Skipping game after {consecutive_failures} failures: "
            f"{matchup_id}, mode={mode.value}, seed={seed}"
        )
        return BenchmarkResult(
            matchup_id=matchup_id,
            mode=mode,
            seed=seed,
            game_index=game_index,
            episode_id="FAILED",
            winner=None,
            metrics=None,  # type: ignore
            red_models={
                "cluer": matchup.red_team.cluer.model_id,
                "guesser_1": matchup.red_team.guesser_1.model_id,
                "guesser_2": matchup.red_team.guesser_2.model_id,
            },
            blue_models={
                "cluer": matchup.blue_team.cluer.model_id,
                "guesser_1": matchup.blue_team.guesser_1.model_id,
                "guesser_2": matchup.blue_team.guesser_2.model_id,
            },
            duration_seconds=0.0,
            error=f"Failed after {consecutive_failures} attempts",
        )

    async def run(self, callback: callable = None) -> list[BenchmarkResult]:
        """
        Run the complete benchmark.

        Args:
            callback: Optional callback(completed, total, result) for progress updates

        Returns:
            List of BenchmarkResult
        """
        self._setup_output_dirs()
        self._save_config()
        self.progress = self._load_progress()
        self.results = self._load_results()

        matchups = generate_matchups(self.config)
        total_games = count_total_games(self.config)

        logger.info(
            f"Starting benchmark '{self.config.name}' with {total_games} games"
        )
        logger.info(
            f"  {len(matchups)} matchups × {len(self.config.game_modes)} modes × "
            f"{len(self.config.seeds)} seeds × {self.config.games_per_config} games"
        )

        if self.progress.completed_games > 0:
            logger.info(
                f"Resuming from {self.progress.completed_games}/{total_games} completed"
            )

        for matchup in matchups:
            matchup_id = _get_matchup_id(matchup)

            for mode in self.config.game_modes:
                for seed in self.config.seeds:
                    for game_idx in range(self.config.games_per_config):
                        # Check if already completed
                        if self.progress.is_completed(
                            matchup_id, mode, seed, game_idx
                        ):
                            continue

                        # Run game
                        result = await self._run_with_retry(
                            matchup, mode, seed, game_idx
                        )

                        if result:
                            self.results.append(result)
                            self._save_result(result)

                            if result.error:
                                self.progress.mark_failed()
                            else:
                                self.progress.mark_completed(
                                    matchup_id, mode, seed, game_idx
                                )

                            self._save_progress()

                            if callback:
                                callback(
                                    self.progress.completed_games,
                                    total_games,
                                    result,
                                )

                            # Log progress
                            pct = (self.progress.completed_games / total_games) * 100
                            logger.info(
                                f"Progress: {self.progress.completed_games}/{total_games} "
                                f"({pct:.1f}%) - Last: {matchup_id}, seed={seed}"
                            )

        logger.info(
            f"Benchmark complete: {self.progress.completed_games} completed, "
            f"{self.progress.failed_games} failed"
        )

        return self.results


async def run_benchmark(config: ExperimentConfig) -> list[BenchmarkResult]:
    """
    Convenience function to run a benchmark experiment.

    Args:
        config: Experiment configuration

    Returns:
        List of BenchmarkResult
    """
    runner = BenchmarkRunner(config)
    return await runner.run()
