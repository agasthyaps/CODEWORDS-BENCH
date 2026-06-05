from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agents.llm import create_provider
from src.benchmark.config import MatchupConfig
from src.core.benchmarking import (
    BenchmarkCondition,
    BenchmarkPenalty,
    HarnessEvent,
    RunManifest,
    build_run_manifest,
    default_benchmark_condition,
)
from src.core.state import AgentStateManager

from .agents.llm_agents import DecryptoCluerLLM, DecryptoGuesserLLM, run_bounded_action
from .benchmark_config import DecryptoExperimentConfig, generate_decrypto_matchups
from .game import create_game
from .models import DecryptoConfig, DecryptoEpisodeRecord, TeamKey
from .orchestrator import run_episode


class DecryptoBenchmarkResult(BaseModel):
    matchup_id: str
    seed: int
    game_index: int
    episode_id: str
    winner: TeamKey | None
    result_reason: str | None
    red_models: dict[str, str]
    blue_models: dict[str, str]
    duration_seconds: float
    scores: dict[str, Any]
    error: str | None = None
    condition_name: str = "human_table_scratchpad"
    run_manifest: RunManifest | None = None
    benchmark_penalties: list[BenchmarkPenalty] = Field(default_factory=list)
    penalty_summary: dict[str, int | float] = Field(default_factory=dict)


class DecryptoBenchmarkProgress(BaseModel):
    experiment_name: str
    started_at: datetime
    total_games: int
    completed_games: int
    failed_games: int
    completed_keys: set[str] = set()

    def key(self, matchup_id: str, seed: int, game_index: int) -> str:
        return f"{matchup_id}|{seed}|{game_index}"

    def is_completed(self, matchup_id: str, seed: int, game_index: int) -> bool:
        return self.key(matchup_id, seed, game_index) in self.completed_keys

    def mark_completed(self, matchup_id: str, seed: int, game_index: int) -> None:
        self.completed_keys.add(self.key(matchup_id, seed, game_index))
        self.completed_games += 1

    def mark_failed(self) -> None:
        self.failed_games += 1


def _matchup_id(matchup: MatchupConfig) -> str:
    red = f"R({matchup.red_team.cluer.name},{matchup.red_team.guesser_1.name},{matchup.red_team.guesser_2.name})"
    blue = f"B({matchup.blue_team.cluer.name},{matchup.blue_team.guesser_1.name},{matchup.blue_team.guesser_2.name})"
    meta = []
    if matchup.pair_key:
        meta.append(f"pair={matchup.pair_key}")
    if matchup.config_type:
        meta.append(f"cfg={matchup.config_type}")
    meta_str = "|".join(meta) if meta else matchup.composition.value
    return f"{meta_str}:{red}vs{blue}"


@dataclass(frozen=True)
class _LLMTeam:
    cluer: DecryptoCluerLLM
    g1: DecryptoGuesserLLM
    g2: DecryptoGuesserLLM


def _build_team(team_key: TeamKey, assignment, temperature: float) -> _LLMTeam:
    # Providers
    cluer_provider = create_provider(
        assignment.cluer.provider,
        assignment.cluer.model_id,
        base_url=assignment.cluer.base_url,
    )
    g1_provider = create_provider(
        assignment.guesser_1.provider,
        assignment.guesser_1.model_id,
        base_url=assignment.guesser_1.base_url,
    )
    g2_provider = create_provider(
        assignment.guesser_2.provider,
        assignment.guesser_2.model_id,
        base_url=assignment.guesser_2.base_url,
    )

    return _LLMTeam(
        cluer=DecryptoCluerLLM(provider=cluer_provider, model_id=assignment.cluer.model_id, temperature=temperature),
        g1=DecryptoGuesserLLM(provider=g1_provider, agent_id=f"{team_key}_guesser_1", team=team_key, temperature=temperature),
        g2=DecryptoGuesserLLM(provider=g2_provider, agent_id=f"{team_key}_guesser_2", team=team_key, temperature=temperature),
    )


def _team_metadata(team_key: TeamKey, assignment: Any) -> dict[str, Any]:
    return {
        "type": "llm",
        "cluer_model": assignment.cluer.model_id,
        "guesser_1_model": assignment.guesser_1.model_id,
        "guesser_2_model": assignment.guesser_2.model_id,
        "agent_models": {
            f"{team_key}_guesser_1": assignment.guesser_1.model_id,
            f"{team_key}_guesser_2": assignment.guesser_2.model_id,
            f"{team_key}_cluer": assignment.cluer.model_id,
        },
    }


def _penalty_summary(penalties: list[BenchmarkPenalty]) -> dict[str, int | float]:
    summary: dict[str, int | float] = {
        "total": len(penalties),
        "points": sum(p.points for p in penalties),
    }
    for penalty in penalties:
        summary[penalty.penalty_type] = int(summary.get(penalty.penalty_type, 0)) + 1
    return summary


def _action_penalties(action: Any, *, episode_id: str, round_number: int) -> list[BenchmarkPenalty]:
    penalties: list[BenchmarkPenalty] = []
    for independent in action.independent:
        if not independent.parse_ok:
            penalties.append(
                BenchmarkPenalty(
                    penalty_type="parse_failure",
                    game_type="decrypto",
                    episode_id=episode_id,
                    round_number=round_number,
                    team=action.team,
                    agent_id=independent.agent_id,
                    description="Independent Decrypto guess failed parsing.",
                    metadata={"kind": action.kind, "parse_error": independent.parse_error},
                )
            )
        if independent.parse_retry_used:
            penalties.append(
                BenchmarkPenalty(
                    penalty_type="repair_prompt",
                    game_type="decrypto",
                    episode_id=episode_id,
                    round_number=round_number,
                    team=action.team,
                    agent_id=independent.agent_id,
                    description="Independent Decrypto guess used a repair prompt.",
                    metadata={"kind": action.kind, "parse_error": independent.parse_error},
                )
            )

    consensus = action.consensus
    if not consensus.parse_ok:
        penalties.append(
            BenchmarkPenalty(
                penalty_type="fallback_action",
                game_type="decrypto",
                episode_id=episode_id,
                round_number=round_number,
                team=action.team,
                agent_id=consensus.captain_id,
                description="Consensus Decrypto fallback guess was used.",
                metadata={"kind": action.kind, "parse_error": consensus.parse_error},
            )
        )
    if consensus.parse_retry_used:
        penalties.append(
            BenchmarkPenalty(
                penalty_type="repair_prompt",
                game_type="decrypto",
                episode_id=episode_id,
                round_number=round_number,
                team=action.team,
                agent_id=consensus.captain_id,
                description="Consensus Decrypto guess used a repair prompt.",
                metadata={"kind": action.kind, "parse_error": consensus.parse_error},
            )
        )
    return penalties


async def run_single_game(
    matchup: MatchupConfig,
    *,
    seed: int,
    game_index: int,
    temperature: float,
    episodes_dir: Path,
    emit_fn: Any | None = None,
    condition: BenchmarkCondition | None = None,
) -> tuple[DecryptoEpisodeRecord, DecryptoBenchmarkResult]:
    decrypto_config = DecryptoConfig(max_rounds=8, seed=seed)
    game_id, actual_seed, keys, code_sequences = create_game(decrypto_config)
    condition = condition or default_benchmark_condition()
    episode_id = f"{seed:04d}-{game_index:02d}-{game_id}"

    red_team = _build_team("red", matchup.red_team, temperature)
    blue_team = _build_team("blue", matchup.blue_team, temperature)
    agent_states = AgentStateManager() if condition.scratchpad_enabled else None
    harness_events: list[HarnessEvent] = []
    benchmark_penalties: list[BenchmarkPenalty] = []

    async def run_cluer_fn(round_inputs, team: TeamKey):
        agent = red_team.cluer if team == "red" else blue_team.cluer
        agent_id = f"{team}_cluer"
        scratchpad = agent_states.get_scratchpad(agent_id) if agent_states else ""
        clue_set, trace_data, scratchpad_add = await agent.generate(
            round_inputs, team, scratchpad
        )
        if agent_states is not None and scratchpad_add:
            agent_states.get_or_create(agent_id).append_to_scratchpad(
                round_inputs.round_number, scratchpad_add
            )
        return clue_set, trace_data

    async def run_action_fn(round_inputs, team: TeamKey, opponent_team: TeamKey, kind: str):
        acting = red_team if team == "red" else blue_team
        g1_id = f"{team}_guesser_1"
        g2_id = f"{team}_guesser_2"
        g1_scratchpad = agent_states.get_scratchpad(g1_id) if agent_states else ""
        g2_scratchpad = agent_states.get_scratchpad(g2_id) if agent_states else ""
        action_log, scratchpad_adds = await run_bounded_action(
            round_inputs,
            team,
            opponent_team,
            kind,
            acting.g1,
            acting.g2,
            g1_scratchpad,
            g2_scratchpad,
        )
        if agent_states is not None:
            for agent_id, addition in scratchpad_adds.items():
                if addition:
                    agent_states.get_or_create(agent_id).append_to_scratchpad(
                        round_inputs.round_number, addition
                    )
        benchmark_penalties.extend(
            _action_penalties(
                action_log,
                episode_id=episode_id,
                round_number=round_inputs.round_number,
            )
        )
        return action_log

    start = time.time()
    metadata = {
        "red_team": _team_metadata("red", matchup.red_team),
        "blue_team": _team_metadata("blue", matchup.blue_team),
        "condition": condition.model_dump(mode="json"),
    }
    manifest = build_run_manifest(
        game_type="decrypto",
        condition=condition,
        models={
            "red_team": matchup.red_team.model_dump(mode="json"),
            "blue_team": matchup.blue_team.model_dump(mode="json"),
        },
        provider_parameters={"temperature": temperature},
        seed_schedule=[seed],
        game_rules_version="decrypto_v1",
    )
    episode = await run_episode(
        config=decrypto_config,
        game_id=game_id,
        keys=keys,
        code_sequences=code_sequences,
        run_cluer_fn=run_cluer_fn,
        run_action_fn=run_action_fn,
        episode_id=episode_id,
        timestamp=datetime.utcnow(),
        metadata=metadata,
        emit_fn=emit_fn,
        run_manifest=manifest,
        harness_events=harness_events,
        benchmark_penalties=benchmark_penalties,
    )
    if agent_states is not None:
        agent_scratchpads = {
            agent_id: agent_state.scratchpad
            for agent_id, agent_state in agent_states.get_all_states().items()
            if agent_state.scratchpad
        }
        episode = episode.model_copy(update={"agent_scratchpads": agent_scratchpads})
    duration = time.time() - start

    episode.save(str(episodes_dir))

    result = DecryptoBenchmarkResult(
        matchup_id=_matchup_id(matchup),
        seed=seed,
        game_index=game_index,
        episode_id=episode.episode_id,
        winner=episode.winner,
        result_reason=episode.result_reason,
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
        scores=episode.scores,
        condition_name=condition.name.value,
        run_manifest=episode.run_manifest,
        benchmark_penalties=episode.benchmark_penalties,
        penalty_summary=_penalty_summary(episode.benchmark_penalties),
    )

    return episode, result


class DecryptoBenchmarkRunner:
    def __init__(self, config: DecryptoExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.episodes_dir = self.output_dir / "episodes"
        self.results_path = self.output_dir / "results.jsonl"
        self.config_path = self.output_dir / "config.json"
        self.progress_path = self.output_dir / "progress.json"
        self.progress: DecryptoBenchmarkProgress | None = None

    def _setup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(
                {
                    "name": self.config.name,
                    "models": [m.model_dump(mode="json") for m in self.config.models],
                    "seeds": self.config.seeds,
                    "condition": self.config.condition.model_dump(mode="json"),
                    "games_per_config": self.config.games_per_config,
                    "temperature": self.config.temperature,
                },
                f,
                indent=2,
            )

    def _load_progress(self, total_games: int) -> DecryptoBenchmarkProgress:
        if self.progress_path.exists():
            with open(self.progress_path, "r") as f:
                data = json.load(f)
            # Restore set
            data["completed_keys"] = set(data.get("completed_keys", []))
            return DecryptoBenchmarkProgress.model_validate(data)

        return DecryptoBenchmarkProgress(
            experiment_name=self.config.name,
            started_at=datetime.utcnow(),
            total_games=total_games,
            completed_games=0,
            failed_games=0,
            completed_keys=set(),
        )

    def _save_progress(self) -> None:
        if self.progress is None:
            return
        data = self.progress.model_dump(mode="json")
        data["completed_keys"] = list(self.progress.completed_keys)
        with open(self.progress_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def run(self, callback: callable | None = None) -> list[DecryptoBenchmarkResult]:
        self._setup()
        matchups = generate_decrypto_matchups(self.config.models)
        results: list[DecryptoBenchmarkResult] = []

        total = len(matchups) * len(self.config.seeds) * self.config.games_per_config
        self.progress = self._load_progress(total)
        completed = self.progress.completed_games

        for matchup in matchups:
            for seed in self.config.seeds:
                for game_idx in range(self.config.games_per_config):
                    matchup_id = _matchup_id(matchup)
                    if self.progress.is_completed(matchup_id, seed, game_idx):
                        continue
                    try:
                        _episode, result = await run_single_game(
                            matchup,
                            seed=seed,
                            game_index=game_idx,
                            temperature=self.config.temperature,
                            episodes_dir=self.episodes_dir,
                            condition=self.config.condition,
                        )
                    except Exception as e:
                        result = DecryptoBenchmarkResult(
                            matchup_id=matchup_id,
                            seed=seed,
                            game_index=game_idx,
                            episode_id="FAILED",
                            winner=None,
                            result_reason=None,
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
                            scores={},
                            error=str(e),
                            condition_name=self.config.condition.name.value,
                        )

                    results.append(result)
                    with open(self.results_path, "a") as f:
                        f.write(json.dumps(result.model_dump(mode="json")) + "\n")

                    if result.error:
                        self.progress.mark_failed()
                    else:
                        self.progress.mark_completed(matchup_id, seed, game_idx)
                    self._save_progress()
                    completed = self.progress.completed_games
                    if callback:
                        callback(completed, total, result)

        return results
