"""Cloud benchmark runner with parallel execution."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.llm import create_provider
from src.benchmark.config import (
    ExperimentConfig,
    ModelConfig,
    TeamComposition,
    generate_matchups,
    count_total_games,
)
from src.benchmark.model_farm import load_model_farm
from src.benchmark.runner import BenchmarkRunner, BenchmarkResult
from src.core.state import AgentStateManager
from src.decrypto.benchmark_config import DecryptoExperimentConfig, generate_decrypto_matchups
from src.decrypto.benchmark_runner import DecryptoBenchmarkRunner, DecryptoBenchmarkResult
from src.engine import GameMode
from src.hanabi.agents.llm_agent import HanabiPlayerLLM
from src.hanabi.models import HanabiConfig
from src.hanabi.orchestrator import run_episode as run_hanabi_episode
from src.hanabi.metrics import compute_episode_metrics as compute_hanabi_metrics

from .analysis import analyze_batch, InterimFinding
from .config import CloudBenchmarkConfig
from .events import BenchmarkEvent, GameTypeProgressData
from .state import BenchmarkState

logger = logging.getLogger(__name__)


class CloudBenchmarkRunner:
    """
    Coordinates parallel benchmark runs across all game types.

    Features:
    - Runs Codenames, Decrypto, and Hanabi in parallel
    - Configurable concurrency per game type
    - Crash recovery via persistent state
    - Interim analysis every N games
    - SSE event streaming for monitoring
    - Graceful pause support
    """

    def __init__(self, config: CloudBenchmarkConfig):
        # load_or_create may return updated config (from saved snapshot when resuming)
        self.state, self.config = BenchmarkState.load_or_create(config)
        self._pause_requested = False
        self._event_queue: asyncio.Queue[BenchmarkEvent] = asyncio.Queue()
        self._start_time: float | None = None

        # Load models
        self._models, _ = load_model_farm("config/models.json")
        self._model_by_id = {m.model_id: m for m in self._models}

        # Filter to requested models
        self._selected_models = [
            m for m in self._models if m.model_id in config.model_ids
        ]
        if len(self._selected_models) < 2 and (config.run_codenames or config.run_decrypto):
            raise ValueError("Need at least 2 models for competitive games")

        # Results storage for interim analysis
        self._codenames_results: list[dict] = []
        self._decrypto_results: list[dict] = []
        self._hanabi_results: list[dict] = []

        # Setup output directory
        self._output_dir = config.get_output_path()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize totals
        self._init_game_totals()

    def _init_game_totals(self) -> None:
        """Calculate and set total games for each type."""
        seeds = self.config.get_seeds()

        if self.config.run_codenames:
            cn_config = ExperimentConfig(
                name="count",
                models=self._selected_models,
                game_modes=[self.config.codenames_mode],
                seeds=seeds,
                games_per_config=1,
            )
            self.state.codenames.total_games = count_total_games(cn_config)

        if self.config.run_decrypto:
            matchups = generate_decrypto_matchups(self._selected_models)
            self.state.decrypto.total_games = len(matchups) * len(seeds)

        if self.config.run_hanabi:
            # One game per model per seed
            self.state.hanabi.total_games = len(self._selected_models) * len(seeds)

    @property
    def event_queue(self) -> asyncio.Queue[BenchmarkEvent]:
        """Queue for SSE events."""
        return self._event_queue

    def request_pause(self) -> None:
        """Signal workers to stop after current games."""
        self._pause_requested = True
        logger.info("Pause requested - workers will stop after current games")

    async def _emit(self, event: BenchmarkEvent) -> None:
        """Emit an event to the queue."""
        await self._event_queue.put(event)

    async def _emit_progress(self) -> None:
        """Emit a progress event."""
        elapsed = time.time() - self._start_time if self._start_time else 0

        codenames_progress = None
        decrypto_progress = None
        hanabi_progress = None

        if self.config.run_codenames:
            codenames_progress = GameTypeProgressData(
                total=self.state.codenames.total_games,
                completed=self.state.codenames.completed_games,
                failed=self.state.codenames.failed_games,
                running=self.state.codenames.running_games,
            )

        if self.config.run_decrypto:
            decrypto_progress = GameTypeProgressData(
                total=self.state.decrypto.total_games,
                completed=self.state.decrypto.completed_games,
                failed=self.state.decrypto.failed_games,
                running=self.state.decrypto.running_games,
            )

        if self.config.run_hanabi:
            hanabi_progress = GameTypeProgressData(
                total=self.state.hanabi.total_games,
                completed=self.state.hanabi.completed_games,
                failed=self.state.hanabi.failed_games,
                running=self.state.hanabi.running_games,
            )

        await self._emit(
            BenchmarkEvent.progress(
                codenames=codenames_progress,
                decrypto=decrypto_progress,
                hanabi=hanabi_progress,
                elapsed_seconds=elapsed,
            )
        )

    async def run(self) -> None:
        """Run all game types in parallel."""
        self._start_time = time.time()
        self.state.status = "running"
        self.state.save()

        logger.info(f"Starting benchmark '{self.config.experiment_name}'")
        logger.info(f"  Codenames: {self.state.codenames.total_games} games")
        logger.info(f"  Decrypto: {self.state.decrypto.total_games} games")
        logger.info(f"  Hanabi: {self.state.hanabi.total_games} games")

        # Save config
        config_path = self._output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(mode="json"), f, indent=2)

        try:
            # Run all game types in parallel
            tasks = []
            if self.config.run_codenames:
                tasks.append(self._run_codenames_batch())
            if self.config.run_decrypto:
                tasks.append(self._run_decrypto_batch())
            if self.config.run_hanabi:
                tasks.append(self._run_hanabi_batch())

            # Add analysis worker
            tasks.append(self._run_analysis_worker())

            # Add progress emitter
            tasks.append(self._run_progress_emitter())

            await asyncio.gather(*tasks)

            # Final state
            if self._pause_requested:
                self.state.status = "paused"
                await self._emit(
                    BenchmarkEvent.benchmark_paused(
                        experiment_name=self.config.experiment_name,
                        completed_games=self.state.total_completed(),
                        remaining_games=self.state.total_remaining(),
                    )
                )
            else:
                self.state.status = "complete"
                elapsed = time.time() - self._start_time
                await self._emit(
                    BenchmarkEvent.benchmark_complete(
                        experiment_name=self.config.experiment_name,
                        total_games=(
                            self.state.codenames.total_games
                            + self.state.decrypto.total_games
                            + self.state.hanabi.total_games
                        ),
                        completed_games=self.state.total_completed(),
                        failed_games=self.state.total_failed(),
                        elapsed_seconds=elapsed,
                        findings_count=self.state.findings_count,
                    )
                )

        except Exception as e:
            self.state.status = "error"
            self.state.last_error = str(e)
            await self._emit(BenchmarkEvent.benchmark_error(str(e), recoverable=True))
            raise

        finally:
            self.state.save()

    async def _run_progress_emitter(self) -> None:
        """Emit progress events periodically."""
        while not self._pause_requested and self.state.status == "running":
            await self._emit_progress()
            await asyncio.sleep(5)  # Emit every 5 seconds

    async def _run_codenames_batch(self) -> None:
        """Run all Codenames games with concurrency control."""
        sem = asyncio.Semaphore(self.config.codenames_concurrency)
        seeds = self.config.get_seeds()

        cn_config = ExperimentConfig(
            name=f"{self.config.experiment_name}_codenames",
            models=self._selected_models,
            game_modes=[self.config.codenames_mode],
            seeds=seeds,
            games_per_config=1,
            temperature=self.config.temperature,
            max_discussion_rounds=self.config.codenames_max_discussion_rounds,
            max_turns=self.config.codenames_max_turns,
            max_retries=self.config.max_retries,
            output_dir=str(self._output_dir),
        )

        matchups = generate_matchups(cn_config)

        async def run_game(matchup, seed, game_idx):
            if self._pause_requested:
                return

            matchup_id = self._get_codenames_matchup_id(matchup)
            if self.state.is_completed("codenames", matchup_id, seed, game_idx):
                return

            async with sem:
                if self._pause_requested:
                    return

                self.state.mark_running("codenames", 1)
                game_id = f"{matchup_id}_{seed}_{game_idx}"

                models = {
                    "red_cluer": matchup.red_team.cluer.model_id,
                    "red_guesser_1": matchup.red_team.guesser_1.model_id,
                    "red_guesser_2": matchup.red_team.guesser_2.model_id,
                    "blue_cluer": matchup.blue_team.cluer.model_id,
                    "blue_guesser_1": matchup.blue_team.guesser_1.model_id,
                    "blue_guesser_2": matchup.blue_team.guesser_2.model_id,
                }

                await self._emit(
                    BenchmarkEvent.game_start(
                        game_type="codenames",
                        game_id=game_id,
                        seed=seed,
                        models=models,
                        matchup_id=matchup_id,
                    )
                )

                try:
                    start_time = time.time()
                    result = await self._run_single_codenames_game(
                        matchup, cn_config, seed, game_idx
                    )
                    duration = time.time() - start_time

                    self.state.mark_completed("codenames", matchup_id, seed, game_idx)
                    self._codenames_results.append(result.model_dump(mode="json"))

                    await self._emit(
                        BenchmarkEvent.game_complete(
                            game_type="codenames",
                            game_id=game_id,
                            seed=seed,
                            result={
                                "winner": result.winner.value if result.winner else None,
                                "episode_id": result.episode_id,
                            },
                            duration_seconds=duration,
                            matchup_id=matchup_id,
                        )
                    )

                except Exception as e:
                    self.state.mark_failed("codenames", str(e))
                    await self._emit(
                        BenchmarkEvent.game_error(
                            game_type="codenames",
                            game_id=game_id,
                            seed=seed,
                            error=str(e),
                            attempt=1,
                            matchup_id=matchup_id,
                        )
                    )

                    if self.state.should_circuit_break(self.config.max_consecutive_failures):
                        logger.error("Circuit breaker triggered - too many failures")
                        self._pause_requested = True

                finally:
                    self.state.mark_running("codenames", -1)
                    self.state.save()

        # Create all tasks
        tasks = []
        for matchup in matchups:
            for seed in seeds:
                tasks.append(run_game(matchup, seed, 0))

        await asyncio.gather(*tasks)

    async def _run_single_codenames_game(
        self, matchup, config: ExperimentConfig, seed: int, game_idx: int
    ) -> BenchmarkResult:
        """Run a single Codenames game."""
        from src.benchmark.runner import run_single_game, _get_matchup_id

        episode, metrics = await run_single_game(matchup, config.game_modes[0], seed, config)

        # Save episode
        episodes_dir = self._output_dir / "codenames" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        episode.save(episodes_dir)

        return BenchmarkResult(
            matchup_id=_get_matchup_id(matchup),
            mode=config.game_modes[0],
            seed=seed,
            game_index=game_idx,
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
            duration_seconds=0,  # Set by caller
        )

    def _get_codenames_matchup_id(self, matchup) -> str:
        """Get matchup ID for Codenames."""
        from src.benchmark.runner import _get_matchup_id
        return _get_matchup_id(matchup)

    async def _run_decrypto_batch(self) -> None:
        """Run all Decrypto games with concurrency control."""
        sem = asyncio.Semaphore(self.config.decrypto_concurrency)
        seeds = self.config.get_seeds()

        matchups = generate_decrypto_matchups(self._selected_models)

        async def run_game(matchup, seed, game_idx):
            if self._pause_requested:
                return

            matchup_id = self._get_decrypto_matchup_id(matchup)
            if self.state.is_completed("decrypto", matchup_id, seed, game_idx):
                return

            async with sem:
                if self._pause_requested:
                    return

                self.state.mark_running("decrypto", 1)
                game_id = f"{matchup_id}_{seed}_{game_idx}"

                models = {
                    "red_cluer": matchup.red_team.cluer.model_id,
                    "red_guesser_1": matchup.red_team.guesser_1.model_id,
                    "red_guesser_2": matchup.red_team.guesser_2.model_id,
                    "blue_cluer": matchup.blue_team.cluer.model_id,
                    "blue_guesser_1": matchup.blue_team.guesser_1.model_id,
                    "blue_guesser_2": matchup.blue_team.guesser_2.model_id,
                }

                await self._emit(
                    BenchmarkEvent.game_start(
                        game_type="decrypto",
                        game_id=game_id,
                        seed=seed,
                        models=models,
                        matchup_id=matchup_id,
                    )
                )

                try:
                    start_time = time.time()
                    result = await self._run_single_decrypto_game(matchup, seed, game_idx)
                    duration = time.time() - start_time

                    self.state.mark_completed("decrypto", matchup_id, seed, game_idx)
                    self._decrypto_results.append(result.model_dump(mode="json"))

                    await self._emit(
                        BenchmarkEvent.game_complete(
                            game_type="decrypto",
                            game_id=game_id,
                            seed=seed,
                            result={
                                "winner": result.winner,
                                "result_reason": result.result_reason,
                                "episode_id": result.episode_id,
                            },
                            duration_seconds=duration,
                            matchup_id=matchup_id,
                        )
                    )

                except Exception as e:
                    self.state.mark_failed("decrypto", str(e))
                    await self._emit(
                        BenchmarkEvent.game_error(
                            game_type="decrypto",
                            game_id=game_id,
                            seed=seed,
                            error=str(e),
                            attempt=1,
                            matchup_id=matchup_id,
                        )
                    )

                    if self.state.should_circuit_break(self.config.max_consecutive_failures):
                        logger.error("Circuit breaker triggered - too many failures")
                        self._pause_requested = True

                finally:
                    self.state.mark_running("decrypto", -1)
                    self.state.save()

        # Create all tasks
        tasks = []
        for matchup in matchups:
            for seed in seeds:
                tasks.append(run_game(matchup, seed, 0))

        await asyncio.gather(*tasks)

    async def _run_single_decrypto_game(
        self, matchup, seed: int, game_idx: int
    ) -> DecryptoBenchmarkResult:
        """Run a single Decrypto game."""
        from src.decrypto.benchmark_runner import run_single_game, _matchup_id

        episodes_dir = self._output_dir / "decrypto" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        _episode, result = await run_single_game(
            matchup,
            seed=seed,
            game_index=game_idx,
            temperature=self.config.temperature,
            episodes_dir=episodes_dir,
        )

        return result

    def _get_decrypto_matchup_id(self, matchup) -> str:
        """Get matchup ID for Decrypto."""
        from src.decrypto.benchmark_runner import _matchup_id
        return _matchup_id(matchup)

    async def _run_hanabi_batch(self) -> None:
        """Run all Hanabi games with concurrency control."""
        sem = asyncio.Semaphore(self.config.hanabi_concurrency)
        seeds = self.config.get_seeds()

        async def run_game(model: ModelConfig, seed: int):
            if self._pause_requested:
                return

            model_combo = model.model_id
            if self.state.is_completed("hanabi", model_combo, seed):
                return

            async with sem:
                if self._pause_requested:
                    return

                self.state.mark_running("hanabi", 1)
                game_id = f"hanabi_{model.name}_{seed}"

                models = {
                    f"player_{i}": model.model_id for i in range(3)
                }

                await self._emit(
                    BenchmarkEvent.game_start(
                        game_type="hanabi",
                        game_id=game_id,
                        seed=seed,
                        models=models,
                        matchup_id=model_combo,
                    )
                )

                try:
                    start_time = time.time()
                    result = await self._run_single_hanabi_game(model, seed)
                    duration = time.time() - start_time

                    self.state.mark_completed("hanabi", model_combo, seed)
                    self._hanabi_results.append(result)

                    await self._emit(
                        BenchmarkEvent.game_complete(
                            game_type="hanabi",
                            game_id=game_id,
                            seed=seed,
                            result={
                                "score": result.get("score", 0),
                                "game_over_reason": result.get("game_over_reason"),
                                "episode_id": result.get("episode_id"),
                            },
                            duration_seconds=duration,
                            matchup_id=model_combo,
                        )
                    )

                except Exception as e:
                    self.state.mark_failed("hanabi", str(e))
                    await self._emit(
                        BenchmarkEvent.game_error(
                            game_type="hanabi",
                            game_id=game_id,
                            seed=seed,
                            error=str(e),
                            attempt=1,
                            matchup_id=model_combo,
                        )
                    )

                    if self.state.should_circuit_break(self.config.max_consecutive_failures):
                        logger.error("Circuit breaker triggered - too many failures")
                        self._pause_requested = True

                finally:
                    self.state.mark_running("hanabi", -1)
                    self.state.save()

        # Create all tasks
        tasks = []
        for model in self._selected_models:
            for seed in seeds:
                tasks.append(run_game(model, seed))

        await asyncio.gather(*tasks)

    async def _run_single_hanabi_game(
        self, model: ModelConfig, seed: int
    ) -> dict[str, Any]:
        """Run a single Hanabi game."""
        # Create players
        provider = create_provider(model.provider, model.model_id, base_url=model.base_url)
        players = [
            HanabiPlayerLLM(
                provider=provider,
                player_id=f"player_{i}",
                temperature=self.config.temperature,
            )
            for i in range(3)
        ]

        config = HanabiConfig(num_players=3, hand_size=5, seed=seed)
        agent_states = AgentStateManager()

        episode = await run_hanabi_episode(
            config=config,
            players=players,
            agent_states=agent_states,
            episode_id=f"hanabi_{model.name}_{seed:04d}",
            metadata={"model": model.model_id, "seed": seed},
        )

        # Save episode
        episodes_dir = self._output_dir / "hanabi" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        episode.save(str(episodes_dir))

        # Compute metrics
        metrics = compute_hanabi_metrics(episode)

        return {
            "model": model.model_id,
            "model_name": model.name,
            "seed": seed,
            "episode_id": episode.episode_id,
            "score": episode.final_score,
            "game_over_reason": episode.game_over_reason,
            "metrics": metrics,
        }

    async def _run_analysis_worker(self) -> None:
        """Run interim analysis when batches complete."""
        batch_size = self.config.interim_analysis_batch_size

        while not self._pause_requested and self.state.status == "running":
            # Check each game type for analysis needs
            await self._check_and_analyze("codenames", self._codenames_results, batch_size)
            await self._check_and_analyze("decrypto", self._decrypto_results, batch_size)
            await self._check_and_analyze("hanabi", self._hanabi_results, batch_size)

            # Check if all done
            if self._all_games_complete():
                break

            await asyncio.sleep(10)  # Check every 10 seconds

    def _all_games_complete(self) -> bool:
        """Check if all games are complete."""
        cn_done = (
            not self.config.run_codenames
            or self.state.codenames.completed_games + self.state.codenames.failed_games
            >= self.state.codenames.total_games
        )
        dc_done = (
            not self.config.run_decrypto
            or self.state.decrypto.completed_games + self.state.decrypto.failed_games
            >= self.state.decrypto.total_games
        )
        hb_done = (
            not self.config.run_hanabi
            or self.state.hanabi.completed_games + self.state.hanabi.failed_games
            >= self.state.hanabi.total_games
        )
        return cn_done and dc_done and hb_done

    async def _check_and_analyze(
        self,
        game_type: str,
        results: list[dict],
        batch_size: int,
    ) -> None:
        """Check if we have enough results for analysis."""
        progress = getattr(self.state, game_type)
        results_since_last = len(results) - progress.last_analyzed_count

        if results_since_last >= batch_size:
            # Get the batch to analyze
            batch_start = progress.last_analyzed_count
            batch = results[batch_start : batch_start + batch_size]
            batch_number = progress.last_analyzed_count // batch_size + 1

            try:
                finding = await analyze_batch(
                    game_type=game_type,  # type: ignore
                    results=batch,
                    batch_number=batch_number,
                    output_dir=self._output_dir,
                )

                progress.last_analyzed_count += len(batch)
                self.state.findings_count += 1
                self.state.save()

                # Emit finding event
                preview = finding.analysis[:200] + "..." if len(finding.analysis) > 200 else finding.analysis
                await self._emit(
                    BenchmarkEvent.finding(
                        finding_id=finding.finding_id,
                        game_type=game_type,  # type: ignore
                        games_analyzed=finding.games_analyzed,
                        preview=preview,
                    )
                )

                logger.info(f"Generated finding {finding.finding_id}")

            except Exception as e:
                logger.warning(f"Failed to analyze {game_type} batch: {e}")
