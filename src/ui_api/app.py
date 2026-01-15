"""FastAPI app for interactive UI."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from src.agents import AgentConfig, CluerAgent, GuesserAgent, create_provider
from src.agents.guesser import parse_discussion_response
from src.benchmark.model_farm import load_model_farm
from src.decrypto.agents.llm_agents import DecryptoCluerLLM, DecryptoGuesserLLM, run_bounded_action
from src.decrypto.game import (
    create_game as create_decrypto_game,
    initial_counters,
    check_winner,
    update_counters_after_round,
)
from src.decrypto.models import (
    ClueSet,
    DecryptoConfig,
    DecryptoEpisodeRecord,
    RoundInputs,
    RoundLog,
    TeamKey,
)
from src.decrypto.orchestrator import _extract_round_state, _state_tag
from src.engine import (
    GameConfig,
    GameMode,
    Phase,
    Team,
    add_discussion_message,
    create_game,
    process_guess,
    process_pass,
    transition_to_guessing,
)
from src.metrics import compute_episode_metrics
from src.runner import TeamAgents
from src.runner.episode import ExtendedEpisodeRecord
from src.runner.orchestrator import run_clue_phase

from .models import (
    BatchStartRequest,
    CodenamesStartRequest,
    DecryptoStartRequest,
    JobStartResponse,
    ReplaySummary,
    TeamRoleConfig,
    TeamSelection,
)
from .stats_analyzer import analyze_and_save
from .storage import (
    ensure_storage,
    list_replays,
    load_replay,
    load_stats_report,
    save_codenames_episode,
    save_decrypto_episode,
    save_batch_log,
)


@dataclass
class Job:
    job_id: str
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    status: Literal["running", "done", "error"] = "running"
    error: str | None = None
    replay_id: str | None = None


app = FastAPI(title="Codewords UI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, Job] = {}
logger = logging.getLogger("ui_api")

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _emit(job: Job, event_type: str, data: dict[str, Any]) -> None:
    await job.queue.put({"event": event_type, "data": data})


async def _close(job: Job) -> None:
    await job.queue.put(None)


async def _emit_new_events(
    job: Job,
    state,
    event_idx: int,
    *,
    delay_ms: int,
) -> int:
    new_events = state.public_transcript[event_idx:]
    for event in new_events:
        payload = {
            "event": event.model_dump() if hasattr(event, "model_dump") else event,
            "revealed": {k: v.value for k, v in state.revealed.items()},
            "current_turn": state.current_turn.value,
            "phase": state.phase.value,
            "current_clue": state.current_clue.model_dump() if state.current_clue else None,
            "guesses_remaining": state.guesses_remaining,
        }
        await _emit(job, "event", payload)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
    return len(state.public_transcript)


async def _sse_stream(job: Job) -> AsyncGenerator[str, None]:
    while True:
        item = await job.queue.get()
        if item is None:
            break
        event = item.get("event", "message")
        data = json.dumps(item.get("data", {}))
        yield f"event: {event}\n" f"data: {data}\n\n"


def _resolve_team_selection(selection: TeamSelection, *, allow_single: bool) -> dict[str, dict[str, str | None]]:
    def _team(t):
        if allow_single:
            return {
                "cluer": t.cluer,
                "guesser_1": t.guesser_1,
                "guesser_2": None,
            }
        return {
            "cluer": t.cluer,
            "guesser_1": t.guesser_1,
            "guesser_2": t.guesser_2 or t.guesser_1,
        }

    return {"red": _team(selection.red), "blue": _team(selection.blue)}


def _load_model_map() -> dict[str, Any]:
    models, _ = load_model_farm("config/models.json")
    return {m.model_id: m for m in models}


def _build_codenames_team(
    model_map: dict[str, Any],
    team: Team,
    config: dict[str, str | None],
    *,
    temperature: float = 0.7,
) -> TeamAgents:
    team_key = team.value.lower()
    cluer_model = model_map[config["cluer"]]
    g1_model = model_map[config["guesser_1"]]
    g2_model = model_map[config["guesser_2"]] if config.get("guesser_2") else None

    cluer_provider = create_provider(
        cluer_model.provider,
        cluer_model.model_id,
        base_url=cluer_model.base_url,
    )
    g1_provider = create_provider(
        g1_model.provider,
        g1_model.model_id,
        base_url=g1_model.base_url,
    )
    g2_provider = (
        create_provider(g2_model.provider, g2_model.model_id, base_url=g2_model.base_url)
        if g2_model
        else None
    )

    cluer = CluerAgent(
        AgentConfig(
            model=cluer_model.model_id,
            role="cluer",
            team=team,
            agent_id=f"{team_key}_cluer",
            temperature=temperature,
        ),
        cluer_provider,
    )
    guesser_1 = GuesserAgent(
        AgentConfig(
            model=g1_model.model_id,
            role="guesser",
            team=team,
            agent_id=f"{team_key}_guesser_1",
            temperature=temperature,
        ),
        g1_provider,
    )
    guesser_2 = None
    if g2_provider:
        guesser_2 = GuesserAgent(
            AgentConfig(
                model=g2_model.model_id,
                role="guesser",
                team=team,
                agent_id=f"{team_key}_guesser_2",
                temperature=temperature,
            ),
            g2_provider,
        )
    return TeamAgents(cluer=cluer, guesser_1=guesser_1, guesser_2=guesser_2)


def _build_decrypto_team(
    model_map: dict[str, Any],
    team_key: TeamKey,
    config: dict[str, str | None],
    *,
    temperature: float = 0.7,
) -> dict[str, Any]:
    cluer_model = model_map[config["cluer"]]
    g1_model = model_map[config["guesser_1"]]
    g2_model = model_map[config["guesser_2"] or config["guesser_1"]]

    cluer_provider = create_provider(
        cluer_model.provider,
        cluer_model.model_id,
        base_url=cluer_model.base_url,
    )
    g1_provider = create_provider(
        g1_model.provider,
        g1_model.model_id,
        base_url=g1_model.base_url,
    )
    g2_provider = create_provider(
        g2_model.provider,
        g2_model.model_id,
        base_url=g2_model.base_url,
    )

    return {
        "cluer": DecryptoCluerLLM(
            provider=cluer_provider,
            model_id=cluer_model.model_id,
            temperature=temperature,
        ),
        "g1": DecryptoGuesserLLM(
            provider=g1_provider,
            agent_id=f"{team_key}_guesser_1",
            team=team_key,
            temperature=temperature,
        ),
        "g2": DecryptoGuesserLLM(
            provider=g2_provider,
            agent_id=f"{team_key}_guesser_2",
            team=team_key,
            temperature=temperature,
        ),
    }


async def _run_codenames_job(job: Job, req: CodenamesStartRequest) -> None:
    try:
        model_map = _load_model_map()
        selection = _resolve_team_selection(
            req.team_selection, allow_single=req.mode == GameMode.SINGLE_GUESSER
        )
        red_team = _build_codenames_team(model_map, Team.RED, selection["red"])
        blue_team = _build_codenames_team(model_map, Team.BLUE, selection["blue"])
        game_config = GameConfig.for_mode(req.mode, seed=req.seed)
        state = create_game(config=game_config)

        await _emit(
            job,
            "init",
            {
                "game_type": "codenames",
                "config": game_config.model_dump(mode="json"),
                "board": state.board.model_dump(mode="json"),
                "starting_team": state.current_turn.value,
            },
        )

        event_idx = 0
        turn_count = 0
        while state.winner is None and turn_count < req.max_turns:
            turn_count += 1
            team_agents = red_team if state.current_turn == Team.RED else blue_team

            # Clue phase
            state, _clue_trace, should_continue = await run_clue_phase(
                team_agents.cluer, state
            )
            event_idx = await _emit_new_events(
                job, state, event_idx, delay_ms=req.event_delay_ms
            )
            if not should_continue:
                continue

            # Discussion phase (stream per message)
            discussion_messages = []
            if req.mode != GameMode.SINGLE_GUESSER and team_agents.guesser_2 is not None:
                consecutive_consensus = 0
                max_messages = req.max_discussion_rounds * 2
                guessers = [team_agents.guesser_1, team_agents.guesser_2]
                for i in range(max_messages):
                    guesser = guessers[i % 2]
                    message, _trace = await guesser.discuss(state, discussion_messages)
                    state = add_discussion_message(
                        state, message.agent_id, message.content
                    )
                    actual_message = state.public_transcript[-1]
                    discussion_messages.append(actual_message)
                    event_idx = await _emit_new_events(
                        job, state, event_idx, delay_ms=req.event_delay_ms
                    )
                    parsed = parse_discussion_response(message.content)
                    if parsed.consensus:
                        consecutive_consensus += 1
                        if consecutive_consensus >= 2:
                            break
                    else:
                        consecutive_consensus = 0

            if state.phase == Phase.DISCUSSION:
                state = transition_to_guessing(state)

            # Guess phase (stream per guess)
            guesses, _trace = await team_agents.primary_guesser.make_guesses(
                state, discussion_messages
            )
            if not guesses:
                state = process_pass(state)
                event_idx = await _emit_new_events(
                    job, state, event_idx, delay_ms=req.event_delay_ms
                )
            else:
                for guess_word in guesses:
                    state, _result, turn_continues = process_guess(guess_word, state)
                    event_idx = await _emit_new_events(
                        job, state, event_idx, delay_ms=req.event_delay_ms
                    )
                    if state.winner is not None:
                        break
                    if not turn_continues:
                        break
                if state.phase == Phase.GUESS and state.winner is None:
                    state = process_pass(state)
                    event_idx = await _emit_new_events(
                        job, state, event_idx, delay_ms=req.event_delay_ms
                    )

            if state.phase == Phase.GAME_OVER:
                break

        episode = ExtendedEpisodeRecord(
            episode_id=str(uuid.uuid4())[:8],
            config=game_config,
            board_seed=state.board_seed,
            board=state.board,
            public_transcript=[e.model_dump() for e in state.public_transcript],
            turn_traces=[],
            winner=state.winner,
            total_turns=turn_count,
            metadata={
                "red_team": selection["red"],
                "blue_team": selection["blue"],
                "max_discussion_rounds": req.max_discussion_rounds,
            },
        )
        path = save_codenames_episode(episode)
        job.replay_id = path.name
        metrics = compute_episode_metrics(episode).model_dump(mode="json")

        await _emit(
            job,
            "done",
            {
                "replay_id": path.name,
                "winner": state.winner.value if state.winner else None,
                "metrics": metrics,
            },
        )

        logger.info("Submitting codenames stats to Opus for %s", path.name)
        report = await analyze_and_save(
            game_type="codenames",
            replay_id=path.name,
            episode=episode.model_dump(mode="json"),
        )
        logger.info("Received codenames stats for %s", path.name)
        await _emit(job, "stats", report)
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await _emit(job, "job_error", {"error": str(exc)})
    finally:
        await _close(job)


async def _run_decrypto_job(job: Job, req: DecryptoStartRequest) -> None:
    try:
        model_map = _load_model_map()
        selection = _resolve_team_selection(req.team_selection, allow_single=False)
        red_team = _build_decrypto_team(model_map, "red", selection["red"])
        blue_team = _build_decrypto_team(model_map, "blue", selection["blue"])

        cfg = DecryptoConfig(seed=req.seed, max_rounds=req.max_rounds)
        game_id, keys, code_sequences = create_decrypto_game(cfg)

        await _emit(
            job,
            "init",
            {
                "game_type": "decrypto",
                "config": cfg.model_dump(mode="json"),
                "keys": keys,
                "game_id": game_id,
            },
        )

        counters = initial_counters()
        history = []
        winner: TeamKey | None = None
        reason: str | None = None

        for r in range(1, cfg.max_rounds + 1):
            base_inputs = RoundInputs(
                game_id=game_id,
                seed=cfg.seed,
                round_number=r,
                keys=keys,
                current_codes={
                    "red": code_sequences["red"][r - 1],
                    "blue": code_sequences["blue"][r - 1],
                },
                history_rounds=tuple(history),
                counters_before=counters,
                public_clues=None,
            )

            (red_clues, red_priv), (blue_clues, blue_priv) = await asyncio.gather(
                red_team["cluer"].generate(base_inputs, "red"),
                blue_team["cluer"].generate(base_inputs, "blue"),
            )

            await _emit(
                job,
                "clue",
                {
                    "team": "red",
                    "clues": list(red_clues.clues),
                    "code": list(code_sequences["red"][r - 1]),
                    "round": r,
                },
            )
            await _emit(
                job,
                "clue",
                {
                    "team": "blue",
                    "clues": list(blue_clues.clues),
                    "code": list(code_sequences["blue"][r - 1]),
                    "round": r,
                },
            )
            if req.event_delay_ms > 0:
                await asyncio.sleep(req.event_delay_ms / 1000)

            round_inputs = base_inputs.model_copy(
                update={"public_clues": {"red": red_clues, "blue": blue_clues}}
            )

            async def _run_action(round_inputs, team: TeamKey, opponent_team: TeamKey, kind: str, discussion_log: list[dict[str, str]]):
                def _emit_discussion(agent_id: str, message: str) -> None:
                    asyncio.create_task(
                        _emit(
                            job,
                            "discussion",
                            {
                                "team": team,
                                "kind": kind,
                                "agent_id": agent_id,
                                "message": message,
                                "round": r,
                            },
                        )
                    )

                acting = red_team if team == "red" else blue_team
                return await run_bounded_action(
                    round_inputs,
                    team,
                    opponent_team,
                    kind,
                    acting["g1"],
                    acting["g2"],
                    discussion_log=discussion_log,
                    max_discussion_turns_per_guesser=req.max_discussion_turns_per_guesser,
                    emit_discussion=_emit_discussion,
                )

            action_tasks: list[tuple[str, str, asyncio.Task, list[dict[str, str]]]] = []
            red_decode_discussion: list[dict[str, str]] = []
            blue_decode_discussion: list[dict[str, str]] = []
            red_intercept_discussion: list[dict[str, str]] = []
            blue_intercept_discussion: list[dict[str, str]] = []
            action_tasks.append(
                ("red", "decode", asyncio.create_task(_run_action(round_inputs, "red", "blue", "decode", red_decode_discussion)), red_decode_discussion)
            )
            action_tasks.append(
                ("blue", "decode", asyncio.create_task(_run_action(round_inputs, "blue", "red", "decode", blue_decode_discussion)), blue_decode_discussion)
            )
            action_tasks.append(
                ("red", "intercept", asyncio.create_task(_run_action(round_inputs, "red", "blue", "intercept", red_intercept_discussion)), red_intercept_discussion)
            )
            action_tasks.append(
                ("blue", "intercept", asyncio.create_task(_run_action(round_inputs, "blue", "red", "intercept", blue_intercept_discussion)), blue_intercept_discussion)
            )

            results: dict[tuple[str, str], Any] = {}
            for team, kind, task, _discussion in action_tasks:
                action = await task
                results[(team, kind)] = action
                await _emit(
                    job,
                    "action",
                    {
                        "team": team,
                        "kind": kind,
                        "action": action.model_dump(mode="json"),
                        "round": r,
                    },
                )
                if req.event_delay_ms > 0:
                    await asyncio.sleep(req.event_delay_ms / 1000)

            true_red = base_inputs.current_codes["red"]
            true_blue = base_inputs.current_codes["blue"]
            red_decode = results[("red", "decode")]
            blue_decode = results[("blue", "decode")]
            red_intercept = results[("red", "intercept")]
            blue_intercept = results[("blue", "intercept")]

            red_decode_correct = bool(red_decode.consensus.guess == true_red)
            blue_decode_correct = bool(blue_decode.consensus.guess == true_blue)
            red_intercept_correct = bool(red_intercept.consensus.guess == true_blue)
            blue_intercept_correct = bool(blue_intercept.consensus.guess == true_red)

            red_decode = red_decode.model_copy(update={"correct": red_decode_correct})
            blue_decode = blue_decode.model_copy(update={"correct": blue_decode_correct})
            red_intercept = red_intercept.model_copy(update={"correct": red_intercept_correct})
            blue_intercept = blue_intercept.model_copy(update={"correct": blue_intercept_correct})

            counters = update_counters_after_round(
                counters,
                red_intercept_correct=red_intercept_correct,
                blue_intercept_correct=blue_intercept_correct,
                red_decode_correct=red_decode_correct,
                blue_decode_correct=blue_decode_correct,
            )

            red_private = dict(red_priv)
            blue_private = dict(blue_priv)
            red_private["discussion"] = {
                "decode": red_decode_discussion,
                "intercept": red_intercept_discussion,
            }
            blue_private["discussion"] = {
                "decode": blue_decode_discussion,
                "intercept": blue_intercept_discussion,
            }
            round_log = RoundLog(
                round_number=r,
                counters_before=base_inputs.counters_before,
                counters_after=counters,
                round_state_at_clue_time={
                    "red": _state_tag(base_inputs.counters_before["red"]),
                    "blue": _state_tag(base_inputs.counters_before["blue"]),
                },
                public_clues={"red": red_clues, "blue": blue_clues},
                reveal_true_codes={"red": true_red, "blue": true_blue},
                actions=(red_decode, blue_decode, red_intercept, blue_intercept),
                round_state=_extract_round_state(
                    (red_decode, blue_decode, red_intercept, blue_intercept)
                ),
                private={"red": red_private, "blue": blue_private},
            )
            history.append(round_log)
            round_payload = round_log.model_dump(mode="json")
            round_payload.pop("private", None)
            round_payload["true_codes"] = {
                "red": list(true_red),
                "blue": list(true_blue),
            }
            round_payload["final_guesses"] = [
                {
                    "team": a.team,
                    "kind": a.kind,
                    "guess": list(a.consensus.guess) if a.consensus.guess else None,
                    "correct": bool(a.correct),
                }
                for a in (red_decode, blue_decode, red_intercept, blue_intercept)
            ]
            await _emit(job, "round", round_payload)
            if req.event_delay_ms > 0:
                await asyncio.sleep(req.event_delay_ms / 1000)

            winner, reason = check_winner(counters, round_number=r, max_rounds=cfg.max_rounds)
            if winner is not None or reason in ("survived", "max_rounds"):
                break

        episode = DecryptoEpisodeRecord(
            episode_id=f"{req.seed:04d}-{game_id}",
            config=cfg,
            game_id=game_id,
            seed=cfg.seed,
            keys=keys,
            code_sequences=code_sequences,
            rounds=tuple(history),
            winner=winner,
            result_reason=reason,  # type: ignore[arg-type]
            scores={},
        )
        from src.decrypto.metrics import compute_episode_scores

        episode = episode.model_copy(update={"scores": compute_episode_scores(episode)})
        replay_id = save_decrypto_episode(episode)
        job.replay_id = replay_id.split("/")[-1]

        await _emit(
            job,
            "done",
            {
                "replay_id": job.replay_id,
                "winner": winner,
                "result_reason": reason,
                "scores": episode.scores,
            },
        )

        logger.info("Submitting decrypto stats to Opus for %s", job.replay_id)
        report = await analyze_and_save(
            game_type="decrypto",
            replay_id=job.replay_id,
            episode=episode.model_dump(mode="json"),
        )
        logger.info("Received decrypto stats for %s", job.replay_id)
        await _emit(job, "stats", report)
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await _emit(job, "job_error", {"error": str(exc)})
    finally:
        await _close(job)


def _sample_team(model_pool: list[str]) -> TeamSelection:
    def _pick() -> str:
        return random.choice(model_pool)

    return TeamSelection(
        red=TeamRoleConfig(cluer=_pick(), guesser_1=_pick(), guesser_2=_pick()),
        blue=TeamRoleConfig(cluer=_pick(), guesser_1=_pick(), guesser_2=_pick()),
    )


async def _run_batch_job(job: Job, req: BatchStartRequest) -> None:
    try:
        results = []
        for idx in range(req.count):
            if req.pinned:
                if req.team_selection is None:
                    raise ValueError("pinned batch requires team_selection")
                selection = req.team_selection
            else:
                if not req.model_pool:
                    raise ValueError("random batch requires model_pool")
                selection = _sample_team(req.model_pool)

            if req.game_type == "codenames":
                sub_job = Job(job_id=f"{job.job_id}-{idx}")
                await _run_codenames_job(
                    sub_job,
                    CodenamesStartRequest(
                        team_selection=selection,
                        mode=req.codenames_mode,
                        max_discussion_rounds=req.max_discussion_rounds,
                        max_turns=req.max_turns,
                        event_delay_ms=req.event_delay_ms,
                    ),
                )
                results.append({"replay_id": sub_job.replay_id, "status": sub_job.status})
            else:
                sub_job = Job(job_id=f"{job.job_id}-{idx}")
                await _run_decrypto_job(
                    sub_job,
                    DecryptoStartRequest(
                        team_selection=selection,
                        seed=req.seed,
                        max_rounds=req.max_rounds,
                        max_discussion_turns_per_guesser=req.max_discussion_turns_per_guesser,
                        event_delay_ms=req.event_delay_ms,
                    ),
                )
                results.append({"replay_id": sub_job.replay_id, "status": sub_job.status})

            await _emit(job, "progress", {"completed": idx + 1, "total": req.count})

        save_batch_log(job.job_id, {"results": results})
        await _emit(job, "done", {"batch_id": job.job_id, "results": results})
        job.status = "done"
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        await _emit(job, "job_error", {"error": str(exc)})
    finally:
        await _close(job)


@app.get("/models")
def get_models() -> list[dict[str, Any]]:
    models, _ = load_model_farm("config/models.json")
    return [m.model_dump(mode="json") for m in models]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/codenames/start", response_model=JobStartResponse)
async def start_codenames(req: CodenamesStartRequest, background: BackgroundTasks) -> JobStartResponse:
    ensure_storage()
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    background.add_task(_run_codenames_job, job, req)
    return JobStartResponse(job_id=job_id)


@app.get("/codenames/{job_id}/events")
async def codenames_events(job_id: str) -> StreamingResponse:
    job = _job(job_id)
    return StreamingResponse(_sse_stream(job), media_type="text/event-stream")


@app.post("/decrypto/start", response_model=JobStartResponse)
async def start_decrypto(req: DecryptoStartRequest, background: BackgroundTasks) -> JobStartResponse:
    ensure_storage()
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    background.add_task(_run_decrypto_job, job, req)
    return JobStartResponse(job_id=job_id)


@app.get("/decrypto/{job_id}/events")
async def decrypto_events(job_id: str) -> StreamingResponse:
    job = _job(job_id)
    return StreamingResponse(_sse_stream(job), media_type="text/event-stream")


@app.get("/replays", response_model=list[ReplaySummary])
def get_replays() -> list[dict[str, Any]]:
    return list_replays()


@app.get("/replays/{game_type}/{replay_id}")
def get_replay(game_type: Literal["codenames", "decrypto"], replay_id: str) -> dict[str, Any]:
    try:
        return load_replay(game_type, replay_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Replay not found")


@app.get("/stats/{replay_id}")
def get_stats(replay_id: str) -> dict[str, Any] | None:
    return load_stats_report(replay_id)


@app.post("/batch/start", response_model=JobStartResponse)
async def start_batch(req: BatchStartRequest, background: BackgroundTasks) -> JobStartResponse:
    ensure_storage()
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    background.add_task(_run_batch_job, job, req)
    return JobStartResponse(job_id=job_id)


@app.get("/batch/{job_id}/events")
async def batch_events(job_id: str) -> StreamingResponse:
    job = _job(job_id)
    return StreamingResponse(_sse_stream(job), media_type="text/event-stream")
