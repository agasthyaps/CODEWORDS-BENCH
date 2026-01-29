"""FastAPI app for interactive UI."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from src.agents import AgentConfig, CluerAgent, GuesserAgent, create_provider
from src.agents.guesser import parse_discussion_response, _top_lists_match
from src.benchmark.model_farm import load_model_farm
from src.core.state import AgentStateManager
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
from src.runner.orchestrator import run_clue_phase, TurnTraces

from .models import (
    BatchStartRequest,
    BenchmarkStartRequest,
    BenchmarkStatusResponse,
    CodenamesStartRequest,
    DecryptoStartRequest,
    FindingDetail,
    FindingSummary,
    GamePeekResponse,
    GameTypeProgressResponse,
    HanabiStartRequest,
    JobStartResponse,
    ReplaySummary,
    RunningGameResponse,
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
    save_hanabi_episode,
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


async def _emit_scratchpad(job: Job, agent_id: str, addition: str, turn: int) -> None:
    """Emit a scratchpad addition event."""
    if addition:
        await _emit(job, "scratchpad", {
            "agent_id": agent_id,
            "addition": addition,
            "turn": turn,
        })


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


def _decrypto_team_metadata(
    model_map: dict[str, Any],
    team_key: TeamKey,
    config: dict[str, str | None],
) -> dict[str, Any]:
    cluer_model = model_map[config["cluer"]].model_id
    g1_model = model_map[config["guesser_1"]].model_id
    g2_key = config["guesser_2"] or config["guesser_1"]
    g2_model = model_map[g2_key].model_id
    return {
        "type": "llm",
        "cluer_model": cluer_model,
        "guesser_1_model": g1_model,
        "guesser_2_model": g2_model,
        "agent_models": {
            f"{team_key}_cluer": cluer_model,
            f"{team_key}_guesser_1": g1_model,
            f"{team_key}_guesser_2": g2_model,
        },
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
        
        # Initialize agent state manager for scratchpads
        agent_states = AgentStateManager()

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
        all_turn_traces: list[TurnTraces] = []
        
        while state.winner is None and turn_count < req.max_turns:
            turn_count += 1
            current_team = state.current_turn
            team_agents = red_team if current_team == Team.RED else blue_team

            # Clue phase with scratchpad
            cluer = team_agents.cluer
            cluer_scratchpad = agent_states.get_scratchpad(cluer.config.agent_id)
            
            clue, clue_trace, clue_scratchpad_add = await cluer.generate_clue(state, cluer_scratchpad)
            
            # Update scratchpad and emit event
            if clue_scratchpad_add:
                agent_state = agent_states.get_or_create(cluer.config.agent_id)
                agent_state.append_to_scratchpad(turn_count, clue_scratchpad_add)
                await _emit_scratchpad(job, cluer.config.agent_id, clue_scratchpad_add, turn_count)
            
            if clue is None:
                # Ghost clue / pass
                from src.engine import end_turn
                state = end_turn(state)
                event_idx = await _emit_new_events(
                    job, state, event_idx, delay_ms=req.event_delay_ms
                )
                continue
            
            # Apply clue to state
            from src.engine import apply_clue
            state = apply_clue(state, clue.word, clue.number)
            event_idx = await _emit_new_events(
                job, state, event_idx, delay_ms=req.event_delay_ms
            )

            # Discussion phase (stream per message) - collect traces
            discussion_messages = []
            discussion_traces = []
            if req.mode != GameMode.SINGLE_GUESSER and team_agents.guesser_2 is not None:
                consecutive_consensus = 0
                previous_top_words: list[str] | None = None
                max_messages = req.max_discussion_rounds * 2
                guessers = [team_agents.guesser_1, team_agents.guesser_2]
                for i in range(max_messages):
                    guesser = guessers[i % 2]
                    guesser_scratchpad = agent_states.get_scratchpad(guesser.config.agent_id)
                    
                    message, trace, disc_scratchpad_add = await guesser.discuss(
                        state, discussion_messages, guesser_scratchpad
                    )
                    discussion_traces.append(trace)
                    
                    # Update scratchpad and emit event
                    if disc_scratchpad_add:
                        agent_state = agent_states.get_or_create(guesser.config.agent_id)
                        agent_state.append_to_scratchpad(turn_count, disc_scratchpad_add)
                        await _emit_scratchpad(job, guesser.config.agent_id, disc_scratchpad_add, turn_count)
                    
                    state = add_discussion_message(
                        state, message.agent_id, message.content
                    )
                    actual_message = state.public_transcript[-1]
                    discussion_messages.append(actual_message)
                    event_idx = await _emit_new_events(
                        job, state, event_idx, delay_ms=req.event_delay_ms
                    )
                    # Check for consensus - requires YES and matching TOP lists
                    parsed = parse_discussion_response(message.content)
                    if parsed.consensus:
                        if consecutive_consensus == 0:
                            # First YES - store the TOP list
                            consecutive_consensus = 1
                            previous_top_words = parsed.top_words
                        else:
                            # Second consecutive YES - check if TOP lists match
                            if _top_lists_match(previous_top_words, parsed.top_words):
                                # Real consensus - both said YES with same words
                                break
                            else:
                                # False consensus - YES but different words, reset
                                consecutive_consensus = 1
                                previous_top_words = parsed.top_words
                    else:
                        consecutive_consensus = 0
                        previous_top_words = None

            if state.phase == Phase.DISCUSSION:
                state = transition_to_guessing(state)

            # Guess phase (stream per guess) - collect trace
            guesser = team_agents.primary_guesser
            guesser_scratchpad = agent_states.get_scratchpad(guesser.config.agent_id)
            
            guesses, guess_trace, guess_scratchpad_add = await guesser.make_guesses(
                state, discussion_messages, guesser_scratchpad
            )
            
            # Update scratchpad and emit event
            if guess_scratchpad_add:
                agent_state = agent_states.get_or_create(guesser.config.agent_id)
                agent_state.append_to_scratchpad(turn_count, guess_scratchpad_add)
                await _emit_scratchpad(job, guesser.config.agent_id, guess_scratchpad_add, turn_count)
            
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

            # Build turn trace
            turn_trace = TurnTraces(
                turn_number=turn_count,
                team=current_team,
                clue_trace=clue_trace,
                discussion_traces=discussion_traces,
                guess_trace=guess_trace,
            )
            all_turn_traces.append(turn_trace)

            if state.phase == Phase.GAME_OVER:
                break

        # Extract final scratchpad contents
        agent_scratchpads = {
            agent_id: agent_state.scratchpad
            for agent_id, agent_state in agent_states.get_all_states().items()
            if agent_state.scratchpad
        }

        episode = ExtendedEpisodeRecord(
            episode_id=str(uuid.uuid4())[:8],
            config=game_config,
            board_seed=state.board_seed,
            board=state.board,
            public_transcript=[e.model_dump() for e in state.public_transcript],
            turn_traces=[t.model_dump(mode="json") for t in all_turn_traces],
            winner=state.winner,
            total_turns=turn_count,
            agent_scratchpads=agent_scratchpads,
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
                "agent_scratchpads": agent_scratchpads,
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
        metadata = {
            "red_team": _decrypto_team_metadata(model_map, "red", selection["red"]),
            "blue_team": _decrypto_team_metadata(model_map, "blue", selection["blue"]),
        }
        
        # Initialize agent state manager for scratchpads
        agent_states = AgentStateManager()

        cfg = DecryptoConfig(seed=req.seed, max_rounds=req.max_rounds)
        game_id, actual_seed, keys, code_sequences = create_decrypto_game(cfg)

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
                seed=actual_seed,
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

            # Clue phase with scratchpads
            red_cluer_scratchpad = agent_states.get_scratchpad("red_cluer")
            blue_cluer_scratchpad = agent_states.get_scratchpad("blue_cluer")
            
            (red_clues, red_priv, red_cluer_add), (blue_clues, blue_priv, blue_cluer_add) = await asyncio.gather(
                red_team["cluer"].generate(base_inputs, "red", red_cluer_scratchpad),
                blue_team["cluer"].generate(base_inputs, "blue", blue_cluer_scratchpad),
            )
            
            # Update cluer scratchpads and emit events
            if red_cluer_add:
                agent_state = agent_states.get_or_create("red_cluer")
                agent_state.append_to_scratchpad(r, red_cluer_add)
                await _emit_scratchpad(job, "red_cluer", red_cluer_add, r)
            
            if blue_cluer_add:
                agent_state = agent_states.get_or_create("blue_cluer")
                agent_state.append_to_scratchpad(r, blue_cluer_add)
                await _emit_scratchpad(job, "blue_cluer", blue_cluer_add, r)

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
                
                # Get scratchpads for guessers
                g1_scratchpad = agent_states.get_scratchpad(f"{team}_guesser_1")
                g2_scratchpad = agent_states.get_scratchpad(f"{team}_guesser_2")
                
                action, scratchpad_adds = await run_bounded_action(
                    round_inputs,
                    team,
                    opponent_team,
                    kind,
                    acting["g1"],
                    acting["g2"],
                    g1_scratchpad,
                    g2_scratchpad,
                    discussion_log=discussion_log,
                    max_discussion_turns_per_guesser=req.max_discussion_turns_per_guesser,
                    emit_discussion=_emit_discussion,
                )
                
                # Update scratchpads and emit events
                for agent_id, addition in scratchpad_adds.items():
                    if addition:
                        agent_state = agent_states.get_or_create(agent_id)
                        agent_state.append_to_scratchpad(r, addition)
                        await _emit_scratchpad(job, agent_id, addition, r)
                
                return action

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

            red_private = dict(red_priv) if isinstance(red_priv, dict) else {}
            blue_private = dict(blue_priv) if isinstance(blue_priv, dict) else {}
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

        # Extract final scratchpad contents
        agent_scratchpads = {
            agent_id: agent_state.scratchpad
            for agent_id, agent_state in agent_states.get_all_states().items()
            if agent_state.scratchpad
        }

        episode = DecryptoEpisodeRecord(
            episode_id=f"{actual_seed:04d}-{game_id}",
            config=cfg,
            game_id=game_id,
            seed=actual_seed,
            keys=keys,
            code_sequences=code_sequences,
            rounds=tuple(history),
            winner=winner,
            result_reason=reason,  # type: ignore[arg-type]
            scores={},
            agent_scratchpads=agent_scratchpads,
            metadata=metadata,
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
                "agent_scratchpads": agent_scratchpads,
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


def _resolve_batch_seeds(req: BatchStartRequest) -> list[int]:
    """
    Resolve seeds based on seed_mode.
    
    - "random": Generate `count` unique random seeds
    - "fixed": Use `fixed_seed` repeated `count` times
    - "list": Use exactly the seeds in `seed_list`
    """
    if req.seed_mode == "random":
        return [random.randint(0, 2**31 - 1) for _ in range(req.count)]
    elif req.seed_mode == "fixed":
        seed = req.fixed_seed if req.fixed_seed is not None else 0
        return [seed] * req.count
    elif req.seed_mode == "list":
        if not req.seed_list:
            raise ValueError("seed_list mode requires non-empty seed_list")
        return req.seed_list
    else:
        raise ValueError(f"Unknown seed_mode: {req.seed_mode}")


async def _run_batch_job(job: Job, req: BatchStartRequest) -> None:
    """
    Run a batch of games with the specified configuration.
    
    Extensible design - to add a new game type:
    1. Add the game_type literal to BatchStartRequest
    2. Add game-specific options to BatchStartRequest
    3. Add the game runner case below
    """
    try:
        seeds = _resolve_batch_seeds(req)
        
        # For "both" mode, we run codenames + decrypto for each seed
        if req.game_type == "both":
            games_per_seed = ["codenames", "decrypto"]
            total_games = len(seeds) * len(games_per_seed)
        else:
            games_per_seed = [req.game_type]
            total_games = len(seeds)
        
        results = []
        game_idx = 0

        for seed in seeds:
            for game_type in games_per_seed:
                game_result: dict[str, Any] = {
                    "game_index": game_idx,
                    "seed": seed,
                    "game_type": game_type,
                }
                
                if game_type == "codenames":
                    sub_job = Job(job_id=f"{job.job_id}-{game_idx}")
                    await _run_codenames_job(
                        sub_job,
                        CodenamesStartRequest(
                            team_selection=req.team_selection,
                            mode=req.codenames_mode,
                            seed=seed,
                            max_discussion_rounds=req.max_discussion_rounds,
                            max_turns=req.max_turns,
                            event_delay_ms=0,
                        ),
                    )
                    game_result.update({
                        "replay_id": sub_job.replay_id,
                        "status": sub_job.status,
                        "error": sub_job.error,
                    })
                    
                elif game_type == "decrypto":
                    sub_job = Job(job_id=f"{job.job_id}-{game_idx}")
                    await _run_decrypto_job(
                        sub_job,
                        DecryptoStartRequest(
                            team_selection=req.team_selection,
                            seed=seed,
                            max_rounds=req.max_rounds,
                            max_discussion_turns_per_guesser=req.max_discussion_turns_per_guesser,
                            event_delay_ms=0,
                        ),
                    )
                    game_result.update({
                        "replay_id": sub_job.replay_id,
                        "status": sub_job.status,
                        "error": sub_job.error,
                    })
                
                elif game_type == "hanabi":
                    if not req.player_models or len(req.player_models) != 3:
                        raise ValueError("Hanabi requires exactly 3 player_models")
                    sub_job = Job(job_id=f"{job.job_id}-{game_idx}")
                    await _run_hanabi_job(
                        sub_job,
                        HanabiStartRequest(
                            player_models=req.player_models,
                            seed=seed,
                            event_delay_ms=0,
                        ),
                    )
                    game_result.update({
                        "replay_id": sub_job.replay_id,
                        "status": sub_job.status,
                        "error": sub_job.error,
                    })
                
                else:
                    raise ValueError(f"Unknown game_type: {game_type}")

                results.append(game_result)
                game_idx += 1
                await _emit(job, "progress", {
                    "completed": game_idx,
                    "total": total_games,
                    "last_result": game_result,
                })

        save_batch_log(job.job_id, {"results": results, "config": req.model_dump(mode="json")})
        await _emit(job, "done", {
            "batch_id": job.job_id,
            "total_games": total_games,
            "results": results,
        })
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


async def _run_hanabi_job(job: Job, req: HanabiStartRequest) -> None:
    """Run a Hanabi game with 3 LLM players."""
    try:
        from src.hanabi.models import HanabiConfig
        from src.hanabi.orchestrator import run_episode
        from src.hanabi.agents.llm_agent import HanabiPlayerLLM
        from src.hanabi.metrics import compute_episode_metrics
        
        model_map = _load_model_map()
        
        if len(req.player_models) != 3:
            raise ValueError(f"Hanabi requires exactly 3 players, got {len(req.player_models)}")
        
        # Initialize agent state manager for scratchpads
        agent_states = AgentStateManager()
        
        # Create players
        players: list[HanabiPlayerLLM] = []
        player_metadata: dict[str, str] = {}
        for i, model_id in enumerate(req.player_models):
            model = model_map[model_id]
            provider = create_provider(
                model.provider,
                model.model_id,
                base_url=model.base_url,
            )
            player_id = f"player_{i + 1}"
            players.append(HanabiPlayerLLM(
                provider=provider,
                player_id=player_id,
                temperature=0.7,
            ))
            player_metadata[player_id] = model.model_id
        
        # Create game config
        config = HanabiConfig(
            num_players=3,
            hand_size=5,
            seed=req.seed,
        )
        
        # Emit function to stream events
        # Note: We filter out "done" events from the orchestrator because we need
        # to emit our own "done" with replay_id after saving the episode
        async def emit_event(event_type: str, data: dict[str, Any]) -> None:
            if event_type == "done":
                # Skip orchestrator's done event - we'll emit our own with replay_id
                return
            await _emit(job, event_type, data)
            if req.event_delay_ms > 0:
                await asyncio.sleep(req.event_delay_ms / 1000)
        
        def sync_emit(event_type: str, data: dict[str, Any]) -> None:
            asyncio.create_task(emit_event(event_type, data))
        
        # Run the episode
        print("[HANABI] Starting run_episode", flush=True)
        episode = await run_episode(
            config=config,
            players=players,
            agent_states=agent_states,
            emit_fn=sync_emit,
            metadata={"player_models": player_metadata},
        )
        print(f"[HANABI] run_episode completed, score={episode.final_score}, reason={episode.game_over_reason}", flush=True)
        
        # Save episode
        replay_id = save_hanabi_episode(episode)
        job.replay_id = replay_id.split("/")[-1]
        print(f"[HANABI] Saved episode as {job.replay_id}", flush=True)
        
        # Emit done event
        metrics = compute_episode_metrics(episode)
        print("[HANABI] Computed metrics, emitting done event", flush=True)
        await _emit(
            job,
            "done",
            {
                "replay_id": job.replay_id,
                "final_score": episode.final_score,
                "game_over_reason": episode.game_over_reason,
                "total_turns": len(episode.turns),
                "metrics": metrics,
                "agent_scratchpads": episode.agent_scratchpads,
            },
        )
        print("[HANABI] done event emitted", flush=True)
        
        # Stats analysis with Opus
        print(f"[HANABI] Submitting stats to Opus for {job.replay_id}", flush=True)
        try:
            report = await analyze_and_save(
                game_type="hanabi",
                replay_id=job.replay_id,
                episode=episode.model_dump(mode="json"),
            )
            print(f"[HANABI] Received stats for {job.replay_id}", flush=True)
            await _emit(job, "stats", report)
        except Exception as stats_exc:
            print(f"[HANABI] Stats analysis failed: {stats_exc}", flush=True)
            import traceback
            traceback.print_exc()
            # Don't fail the job, just skip stats
        job.status = "done"
        
    except Exception as exc:
        print(f"[HANABI] Job failed with exception: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        job.status = "error"
        job.error = str(exc)
        await _emit(job, "job_error", {"error": str(exc)})
    finally:
        await _close(job)


@app.post("/hanabi/start", response_model=JobStartResponse)
async def start_hanabi(req: HanabiStartRequest, background: BackgroundTasks) -> JobStartResponse:
    ensure_storage()
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id=job_id)
    _jobs[job_id] = job
    background.add_task(_run_hanabi_job, job, req)
    return JobStartResponse(job_id=job_id)


@app.get("/hanabi/{job_id}/events")
async def hanabi_events(job_id: str) -> StreamingResponse:
    job = _job(job_id)
    return StreamingResponse(_sse_stream(job), media_type="text/event-stream")


@app.get("/replays", response_model=list[ReplaySummary])
def get_replays() -> list[dict[str, Any]]:
    return list_replays()


@app.get("/replays/{game_type}/{replay_id}")
def get_replay(game_type: Literal["codenames", "decrypto", "hanabi"], replay_id: str) -> dict[str, Any]:
    try:
        return load_replay(game_type, replay_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Replay not found")


@app.get("/stats/{replay_id}")
def get_stats(replay_id: str) -> dict[str, Any] | None:
    return load_stats_report(replay_id)


# ============================================================
# Leaderboard endpoints
# ============================================================


@app.get("/leaderboard")
def get_leaderboard() -> dict[str, Any]:
    """Return current leaderboard data, auto-generating if missing or stale."""
    from .leaderboard_builder import build_leaderboard, load_leaderboard, save_leaderboard, _benchmark_results_dir

    bench_dir = _benchmark_results_dir()
    print(f"[LEADERBOARD] GET - dir={bench_dir}, exists={bench_dir.exists()}")

    data = load_leaderboard()
    if data:
        print(f"[LEADERBOARD] Loaded from disk: {data.total_episodes}, models={len(data.overall_rankings)}")

    # Rebuild if missing or has 0 episodes (might be stale)
    if data is None or sum(data.total_episodes.values()) == 0:
        print("[LEADERBOARD] Rebuilding (missing or empty)...")
        data = build_leaderboard()
        save_leaderboard(data)
        print(f"[LEADERBOARD] Built: {data.total_episodes}, models={len(data.overall_rankings)}")

    return data.model_dump(mode="json")


@app.post("/leaderboard/refresh")
def refresh_leaderboard() -> dict[str, Any]:
    """Recompute leaderboard synchronously and return result."""
    from .leaderboard_builder import build_leaderboard, save_leaderboard, scan_all_episodes, _benchmark_results_dir

    bench_dir = _benchmark_results_dir()
    print(f"[LEADERBOARD] REFRESH - dir={bench_dir}, exists={bench_dir.exists()}")

    # Log what we find
    if bench_dir.exists():
        contents = list(bench_dir.iterdir())
        print(f"[LEADERBOARD] Contents: {[c.name for c in contents[:15]]}")
        # Check structure of each subdir
        for c in contents[:10]:
            if c.is_dir() and c.name not in ('lost+found', 'sessions'):
                subcontents = list(c.iterdir())[:10]
                print(f"[LEADERBOARD]   {c.name}/: {[s.name for s in subcontents]}")
                # Check for episodes subdir
                eps_dir = c / "episodes"
                if eps_dir.exists():
                    eps_count = len(list(eps_dir.glob("*.json")))
                    print(f"[LEADERBOARD]     -> episodes/: {eps_count} files")
                # Check for game type subdirs with episodes
                for game in ("codenames", "decrypto", "hanabi"):
                    game_eps_dir = c / game / "episodes"
                    if game_eps_dir.exists():
                        game_count = len(list(game_eps_dir.glob("*.json")))
                        print(f"[LEADERBOARD]     -> {game}/episodes/: {game_count} files")

        # Also check sessions dir
        sessions_dir = bench_dir / "sessions"
        if sessions_dir.exists():
            print(f"[LEADERBOARD] sessions/ exists: {[d.name for d in sessions_dir.iterdir()]}")
        else:
            print(f"[LEADERBOARD] sessions/ does NOT exist")

    # Scan and log
    episodes = scan_all_episodes()
    print(f"[LEADERBOARD] Scanned: codenames={len(episodes['codenames'])}, decrypto={len(episodes['decrypto'])}, hanabi={len(episodes['hanabi'])}")

    data = build_leaderboard()
    save_leaderboard(data)

    print(f"[LEADERBOARD] Result: {len(data.overall_rankings)} models, {sum(data.total_episodes.values())} games")

    return {
        "status": "refreshed",
        "total_episodes": data.total_episodes,
        "models_count": len(data.overall_rankings),
    }


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


# =============================================================================
# Cloud Benchmark Endpoints
# =============================================================================

from src.cloud_benchmark import (
    CloudBenchmarkConfig,
    CloudBenchmarkRunner,
    BenchmarkState,
)
from src.cloud_benchmark.analysis import list_findings, load_finding

# Global runner instance
_benchmark_runner: CloudBenchmarkRunner | None = None
_benchmark_task: asyncio.Task | None = None


@app.post("/benchmark/start")
async def start_benchmark(
    req: BenchmarkStartRequest, background: BackgroundTasks
) -> dict[str, Any]:
    """Start or resume a cloud benchmark run."""
    global _benchmark_runner, _benchmark_task

    # Check if already running in memory
    if _benchmark_runner and _benchmark_task and not _benchmark_task.done():
        raise HTTPException(409, "Benchmark already running")

    # Check existing state - if state says "running" but no in-memory runner,
    # this was interrupted (crash/deploy) and we should allow resuming
    existing = BenchmarkState.load(req.experiment_name)
    if existing and existing.status == "running":
        # Mark as interrupted so it can be resumed
        existing.status = "paused"
        existing.last_error = "Interrupted (server restart detected)"
        existing.save()
        logger.info(f"Marked interrupted experiment {req.experiment_name} as paused")

    # Create config
    config = CloudBenchmarkConfig(
        experiment_name=req.experiment_name,
        model_ids=req.model_ids,
        seed_count=req.seed_count,
        seed_list=req.seed_list,
        run_codenames=req.run_codenames,
        run_decrypto=req.run_decrypto,
        run_hanabi=req.run_hanabi,
        codenames_concurrency=req.codenames_concurrency,
        decrypto_concurrency=req.decrypto_concurrency,
        hanabi_concurrency=req.hanabi_concurrency,
        codenames_mode=req.codenames_mode,
        codenames_max_turns=req.codenames_max_turns,
        codenames_max_discussion_rounds=req.codenames_max_discussion_rounds,
        decrypto_max_rounds=req.decrypto_max_rounds,
        decrypto_max_discussion_turns=req.decrypto_max_discussion_turns,
        interim_analysis_batch_size=req.interim_analysis_batch_size,
        max_retries=req.max_retries,
        temperature=req.temperature,
    )

    # Create runner
    _benchmark_runner = CloudBenchmarkRunner(config)

    # Start in background
    async def run_benchmark():
        try:
            await _benchmark_runner.run()
        except Exception as e:
            logger.exception(f"Benchmark failed: {e}")

    _benchmark_task = asyncio.create_task(run_benchmark())

    return {
        "experiment_name": req.experiment_name,
        "status": "started",
        "codenames_total": _benchmark_runner.state.codenames.total_games,
        "decrypto_total": _benchmark_runner.state.decrypto.total_games,
        "hanabi_total": _benchmark_runner.state.hanabi.total_games,
    }


@app.get("/benchmark/status")
def benchmark_status() -> BenchmarkStatusResponse:
    """Get current benchmark status."""
    if not _benchmark_runner:
        # Check if there's an active task that's still running
        if _benchmark_task and not _benchmark_task.done():
            # Runner was cleared but task still running - shouldn't happen
            return BenchmarkStatusResponse(status="error", last_error="Runner state inconsistent")
        return BenchmarkStatusResponse(status="idle")

    # Check if task is actually still running
    if _benchmark_task and _benchmark_task.done():
        # Task finished but runner still exists - check for exceptions
        try:
            _benchmark_task.result()
        except Exception as e:
            if _benchmark_runner.state.status == "running":
                _benchmark_runner.state.status = "error"
                _benchmark_runner.state.last_error = str(e)
                _benchmark_runner.state.save()

    state = _benchmark_runner.state

    codenames_progress = None
    decrypto_progress = None
    hanabi_progress = None

    if _benchmark_runner.config.run_codenames:
        codenames_progress = GameTypeProgressResponse(
            total=state.codenames.total_games,
            completed=state.codenames.completed_games,
            failed=state.codenames.failed_games,
            running=state.codenames.running_games,
        )

    if _benchmark_runner.config.run_decrypto:
        decrypto_progress = GameTypeProgressResponse(
            total=state.decrypto.total_games,
            completed=state.decrypto.completed_games,
            failed=state.decrypto.failed_games,
            running=state.decrypto.running_games,
        )

    if _benchmark_runner.config.run_hanabi:
        hanabi_progress = GameTypeProgressResponse(
            total=state.hanabi.total_games,
            completed=state.hanabi.completed_games,
            failed=state.hanabi.failed_games,
            running=state.hanabi.running_games,
        )

    return BenchmarkStatusResponse(
        status=state.status,
        experiment_name=state.experiment_name,
        started_at=state.started_at.isoformat() if state.started_at else None,
        codenames=codenames_progress,
        decrypto=decrypto_progress,
        hanabi=hanabi_progress,
        findings_count=state.findings_count,
        last_error=state.last_error,
    )


@app.get("/benchmark/events")
async def benchmark_events() -> StreamingResponse:
    """SSE stream of benchmark events."""
    if not _benchmark_runner:
        raise HTTPException(404, "No benchmark running")

    async def event_generator():
        queue = _benchmark_runner.event_queue
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                data = event.model_dump(mode="json")
                yield f"event: {event.event_type.value}\ndata: {json.dumps(data)}\n\n"

                # Stop on terminal events
                if event.event_type.value in ("benchmark_complete", "benchmark_paused", "benchmark_error"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f": keepalive\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/benchmark/pause")
def pause_benchmark() -> dict[str, str]:
    """Request graceful pause of the benchmark."""
    if not _benchmark_runner:
        raise HTTPException(404, "No benchmark running")

    _benchmark_runner.request_pause()
    return {"status": "pause_requested"}


@app.post("/benchmark/cancel")
async def cancel_benchmark(experiment_name: str | None = None) -> dict[str, str]:
    """Cancel the running benchmark immediately or force-stop a stale experiment."""
    global _benchmark_runner, _benchmark_task

    # If we have an in-memory runner, cancel it
    if _benchmark_runner and _benchmark_task:
        # Request pause to stop workers
        _benchmark_runner.request_pause()

        # Cancel the task
        if not _benchmark_task.done():
            _benchmark_task.cancel()
            try:
                await _benchmark_task
            except asyncio.CancelledError:
                pass

        # Update state
        _benchmark_runner.state.status = "cancelled"
        _benchmark_runner.state.save()

        # Clear the runner
        _benchmark_runner = None
        _benchmark_task = None

        return {"status": "cancelled"}

    # No in-memory runner - force-stop via state file if experiment_name provided
    if experiment_name:
        existing = BenchmarkState.load(experiment_name)
        if existing:
            existing.status = "cancelled"
            existing.last_error = "Force-stopped by user"
            existing.save()
            return {"status": "cancelled", "experiment_name": experiment_name}
        raise HTTPException(404, f"Experiment '{experiment_name}' not found")

    raise HTTPException(404, "No benchmark running and no experiment_name provided")


@app.post("/benchmark/force-stop/{experiment_name}")
async def force_stop_benchmark(experiment_name: str) -> dict[str, str]:
    """Force-stop an experiment that appears stuck or falsely running."""
    global _benchmark_runner, _benchmark_task

    # If the in-memory runner is for this experiment, cancel it
    if _benchmark_runner and _benchmark_runner.state.experiment_name == experiment_name:
        if _benchmark_task and not _benchmark_task.done():
            _benchmark_runner.request_pause()
            _benchmark_task.cancel()
            try:
                await _benchmark_task
            except asyncio.CancelledError:
                pass
        _benchmark_runner.state.status = "cancelled"
        _benchmark_runner.state.last_error = "Force-stopped by user"
        _benchmark_runner.state.save()
        _benchmark_runner = None
        _benchmark_task = None
        return {"status": "cancelled", "experiment_name": experiment_name}

    # Otherwise, just update the state file
    existing = BenchmarkState.load(experiment_name)
    if not existing:
        raise HTTPException(404, f"Experiment '{experiment_name}' not found")

    existing.status = "cancelled"
    existing.last_error = "Force-stopped by user"
    existing.save()
    return {"status": "cancelled", "experiment_name": experiment_name}


@app.get("/benchmark/download/{experiment_name}")
async def download_benchmark_results(experiment_name: str) -> FileResponse:
    """Download benchmark results as a zip file."""
    import shutil
    import tempfile
    from src.cloud_benchmark.config import get_data_dir

    data_dir = get_data_dir()
    exp_dir = data_dir / experiment_name

    if not exp_dir.exists():
        raise HTTPException(404, f"Experiment not found: {experiment_name}")

    # Create temp zip file
    temp_dir = tempfile.mkdtemp()
    zip_base = Path(temp_dir) / experiment_name
    zip_path = shutil.make_archive(str(zip_base), "zip", exp_dir)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{experiment_name}.zip",
        background=BackgroundTasks(),  # Clean up happens after response
    )


@app.get("/benchmark/findings")
def get_benchmark_findings(experiment_name: str | None = None) -> list[FindingSummary]:
    """List all interim findings for a benchmark.

    If experiment_name is provided, loads findings for that experiment.
    Otherwise uses the currently running benchmark.
    """
    from src.cloud_benchmark.config import get_data_dir

    if experiment_name:
        output_dir = get_data_dir() / experiment_name
    elif _benchmark_runner:
        output_dir = _benchmark_runner.config.get_output_path()
    else:
        return []

    if not output_dir.exists():
        return []

    findings = list_findings(output_dir)

    return [
        FindingSummary(
            finding_id=f.finding_id,
            game_type=f.game_type,
            batch_number=f.batch_number,
            games_analyzed=f.games_analyzed,
            timestamp=f.timestamp.isoformat(),
            preview=f.analysis[:200] + "..." if len(f.analysis) > 200 else f.analysis,
        )
        for f in findings
    ]


@app.get("/benchmark/findings/{finding_id}")
def get_benchmark_finding(finding_id: str, experiment_name: str | None = None) -> FindingDetail:
    """Get a specific finding by ID.

    If experiment_name is provided, loads from that experiment.
    Otherwise uses the currently running benchmark.
    """
    from src.cloud_benchmark.config import get_data_dir

    if experiment_name:
        output_dir = get_data_dir() / experiment_name
    elif _benchmark_runner:
        output_dir = _benchmark_runner.config.get_output_path()
    else:
        raise HTTPException(404, "No benchmark running and no experiment_name provided")

    finding = load_finding(output_dir, finding_id)

    if not finding:
        raise HTTPException(404, f"Finding not found: {finding_id}")

    return FindingDetail(
        finding_id=finding.finding_id,
        game_type=finding.game_type,
        batch_number=finding.batch_number,
        games_analyzed=finding.games_analyzed,
        timestamp=finding.timestamp.isoformat(),
        summary_metrics=finding.summary_metrics,
        analysis=finding.analysis,
        model=finding.model,
        usage=finding.usage,
    )


@app.get("/benchmark/running-games")
def get_running_games() -> list[RunningGameResponse]:
    """List all currently running games with their durations."""
    if not _benchmark_runner:
        return []

    now = datetime.utcnow()
    running_games = _benchmark_runner.get_running_games()

    return [
        RunningGameResponse(
            game_id=g.game_id,
            game_type=g.game_type,
            matchup_id=g.matchup_id,
            seed=g.seed,
            models=g.models,
            started_at=g.started_at.isoformat(),
            duration_seconds=(now - g.started_at).total_seconds(),
            current_turn=g.current_turn,
        )
        for g in running_games
    ]


@app.get("/benchmark/game-peek")
def peek_game(game_id: str) -> GamePeekResponse:
    """Get live state of a running game (current turn, recent transcript, scratchpads)."""
    if not _benchmark_runner:
        raise HTTPException(404, "No benchmark running")

    # Get running game info
    running_games = {g.game_id: g for g in _benchmark_runner.get_running_games()}
    game_info = running_games.get(game_id)

    if not game_info:
        # Log available games for debugging
        available = list(running_games.keys())[:5]
        logger.warning(f"Game not found: {game_id!r}, available: {available}")
        raise HTTPException(404, f"Game not found: {game_id}")

    # Get live state
    live_state = _benchmark_runner.get_live_game_state(game_id)

    now = datetime.utcnow()
    duration = (now - game_info.started_at).total_seconds()

    # Stale threshold: 5 minutes (games can legitimately take a while per turn)
    stale_threshold_seconds = 300

    if live_state and (live_state.recent_transcript or live_state.agent_scratchpads):
        # We have actual live state data
        stale_warning = (now - live_state.last_update).total_seconds() > stale_threshold_seconds
        return GamePeekResponse(
            game_id=game_id,
            game_type=game_info.game_type,
            current_turn=live_state.current_turn,
            recent_transcript=live_state.recent_transcript,
            agent_scratchpads=live_state.agent_scratchpads,
            started_at=game_info.started_at.isoformat(),
            duration_seconds=duration,
            last_update=live_state.last_update.isoformat(),
            stale_warning=stale_warning,
        )
    else:
        # No live state captured - show stale warning only if running > 5 min
        stale_warning = duration > stale_threshold_seconds
        return GamePeekResponse(
            game_id=game_id,
            game_type=game_info.game_type,
            current_turn=game_info.current_turn,
            recent_transcript=[],
            agent_scratchpads={},
            started_at=game_info.started_at.isoformat(),
            duration_seconds=duration,
            last_update=game_info.started_at.isoformat(),
            stale_warning=stale_warning,
        )


@app.post("/benchmark/restart-game")
async def restart_game(game_id: str) -> dict[str, Any]:
    """Cancel a hung game and re-queue it for execution."""
    if not _benchmark_runner:
        raise HTTPException(404, "No benchmark running")

    success = await _benchmark_runner.restart_game(game_id)

    if not success:
        raise HTTPException(400, f"Could not restart game: {game_id}")

    return {
        "status": "restarted",
        "game_id": game_id,
        "message": "Game cancelled and re-queued for execution",
    }


@app.get("/benchmark/experiments")
def list_experiments() -> list[dict[str, Any]]:
    """List all available experiments (for resuming)."""
    from src.cloud_benchmark.config import get_data_dir

    data_dir = get_data_dir()
    if not data_dir.exists():
        return []

    experiments = []
    for exp_dir in data_dir.iterdir():
        if exp_dir.is_dir():
            state_path = exp_dir / "benchmark_state.json"
            if state_path.exists():
                state = BenchmarkState.load(exp_dir.name)
                if state:
                    experiments.append({
                        "experiment_name": state.experiment_name,
                        "status": state.status,
                        "started_at": state.started_at.isoformat() if state.started_at else None,
                        "total_completed": state.total_completed(),
                        "total_failed": state.total_failed(),
                        "findings_count": state.findings_count,
                    })

    return sorted(experiments, key=lambda x: x.get("started_at", ""), reverse=True)


# =============================================================================
# Static File Serving (Production)
# =============================================================================

# Serve built UI in production (must be last to not override API routes)
_ui_dist = Path(__file__).parent.parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=_ui_dist, html=True), name="ui")
