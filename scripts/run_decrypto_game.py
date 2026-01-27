#!/usr/bin/env python3
"""Run a single Decrypto game with LLM agents."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from src.agents.llm import create_provider
from src.core.state import AgentStateManager
from src.decrypto.agents.llm_agents import DecryptoCluerLLM, DecryptoGuesserLLM, run_bounded_action
from src.decrypto.game import check_winner, create_game, initial_counters, update_counters_after_round
from src.decrypto.metrics import compute_episode_scores
from src.decrypto.models import ClueSet, DecryptoConfig, DecryptoEpisodeRecord, RoundInputs, TeamKey


class Colors:
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def tcol(team: TeamKey) -> str:
    return Colors.RED if team == "red" else Colors.BLUE


def print_scratchpad_addition(agent_id: str, addition: str, team: TeamKey):
    """Print when an agent adds to their scratchpad."""
    color = tcol(team)
    # Truncate if too long
    if len(addition) > 150:
        addition = addition[:150] + "..."
    print(f"{Colors.CYAN}  📝 [{agent_id} scratchpad]: {addition}{Colors.RESET}")


def print_all_scratchpads(agent_states: AgentStateManager):
    """Print all agent scratchpads at end of game."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}AGENT SCRATCHPADS{Colors.RESET}")
    print(f"{'=' * 60}")
    
    all_states = agent_states.get_all_states()
    
    if not any(s.scratchpad for s in all_states.values()):
        print(f"{Colors.DIM}  (no agents used scratchpads){Colors.RESET}")
        return
    
    for team in ["red", "blue"]:
        color = Colors.RED if team == "red" else Colors.BLUE
        print(f"\n{color}{Colors.BOLD}{team.upper()} TEAM:{Colors.RESET}")
        
        team_agents = [
            f"{team}_cluer",
            f"{team}_guesser_1",
            f"{team}_guesser_2",
        ]
        
        for agent_id in team_agents:
            state = all_states.get(agent_id)
            if state and state.scratchpad:
                print(f"{color}  [{agent_id}]:{Colors.RESET}")
                for line in state.scratchpad.split("\n"):
                    print(f"{Colors.CYAN}    {line}{Colors.RESET}")
            else:
                print(f"{color}  [{agent_id}]: {Colors.DIM}(empty){Colors.RESET}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Decrypto game")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--show-keys", action="store_true", help="Print both teams' keys (debug)")
    parser.add_argument("--show-annotations", action="store_true", help="Print cluer private annotations (debug)")
    parser.add_argument("--no-scratchpads", action="store_true", help="Hide scratchpad output")

    parser.add_argument("--red-cluer", required=True)
    parser.add_argument("--red-guesser-1", required=True)
    parser.add_argument("--red-guesser-2", required=True)
    parser.add_argument("--blue-cluer", required=True)
    parser.add_argument("--blue-guesser-1", required=True)
    parser.add_argument("--blue-guesser-2", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = DecryptoConfig(seed=args.seed, max_rounds=args.max_rounds)
    game_id, actual_seed, keys, code_sequences = create_game(cfg)
    
    # Initialize agent state manager for scratchpads
    agent_states = AgentStateManager()
    show_scratchpads = not args.no_scratchpads

    def _p(x) -> None:
        if not args.quiet:
            print(x)

    def _team(team: TeamKey, cluer_id: str, g1_id: str, g2_id: str):
        return {
            "cluer": DecryptoCluerLLM(
                provider=create_provider("openrouter", cluer_id),
                model_id=cluer_id,
                temperature=args.temperature,
            ),
            "g1": DecryptoGuesserLLM(
                provider=create_provider("openrouter", g1_id),
                agent_id=f"{team}_guesser_1",
                team=team,
                temperature=args.temperature,
            ),
            "g2": DecryptoGuesserLLM(
                provider=create_provider("openrouter", g2_id),
                agent_id=f"{team}_guesser_2",
                team=team,
                temperature=args.temperature,
            ),
        }

    red = _team("red", args.red_cluer, args.red_guesser_1, args.red_guesser_2)
    blue = _team("blue", args.blue_cluer, args.blue_guesser_1, args.blue_guesser_2)

    _p(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    _p(f"{Colors.BOLD}DECRYPTO GAME - Episode decrypto-{actual_seed}-{game_id}{Colors.RESET}")
    _p(f"Seed: {actual_seed} | Max rounds: {args.max_rounds}")
    _p(f"{Colors.RED}Red:{Colors.RESET} cluer={args.red_cluer} g1={args.red_guesser_1} g2={args.red_guesser_2}")
    _p(f"{Colors.BLUE}Blue:{Colors.RESET} cluer={args.blue_cluer} g1={args.blue_guesser_1} g2={args.blue_guesser_2}")
    if args.show_keys and not args.quiet:
        _p(f"{Colors.RED}Red key:{Colors.RESET} {', '.join(keys['red'])}")
        _p(f"{Colors.BLUE}Blue key:{Colors.RESET} {', '.join(keys['blue'])}")
    _p(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

    counters = initial_counters()
    round_logs = []

    for r in range(1, cfg.max_rounds + 1):
        _p(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        _p(f"{Colors.BOLD}ROUND {r}{Colors.RESET}")
        _p(f"{Colors.BOLD}{'─' * 60}{Colors.RESET}")

        base_inputs = RoundInputs(
            game_id=game_id,
            seed=actual_seed,
            round_number=r,
            keys=keys,
            current_codes={"red": code_sequences["red"][r - 1], "blue": code_sequences["blue"][r - 1]},
            history_rounds=tuple(round_logs),
            counters_before=counters,
            public_clues=None,
        )

        # Clue phase (simultaneous) with scratchpads
        red_cluer_scratchpad = agent_states.get_scratchpad("red_cluer")
        blue_cluer_scratchpad = agent_states.get_scratchpad("blue_cluer")
        
        (red_clues, red_priv, red_cluer_add), (blue_clues, blue_priv, blue_cluer_add) = await asyncio.gather(
            red["cluer"].generate(base_inputs, "red", red_cluer_scratchpad),
            blue["cluer"].generate(base_inputs, "blue", blue_cluer_scratchpad),
        )
        
        # Update cluer scratchpads
        if red_cluer_add:
            state = agent_states.get_or_create("red_cluer")
            state.append_to_scratchpad(r, red_cluer_add)
            if show_scratchpads and not args.quiet:
                print_scratchpad_addition("red_cluer", red_cluer_add, "red")
        
        if blue_cluer_add:
            state = agent_states.get_or_create("blue_cluer")
            state.append_to_scratchpad(r, blue_cluer_add)
            if show_scratchpads and not args.quiet:
                print_scratchpad_addition("blue_cluer", blue_cluer_add, "blue")
        
        if not args.quiet:
            _p(f"{Colors.RED}{Colors.BOLD}[RED CLUES]{Colors.RESET}  {' | '.join(red_clues.clues)}")
            _p(f"{Colors.BLUE}{Colors.BOLD}[BLUE CLUES]{Colors.RESET} {' | '.join(blue_clues.clues)}")
            if args.show_annotations:
                _p(f"{Colors.RED}  (red annotations){Colors.RESET} {red_priv}")
                _p(f"{Colors.BLUE}  (blue annotations){Colors.RESET} {blue_priv}")

        round_inputs = base_inputs.model_copy(update={"public_clues": {"red": red_clues, "blue": blue_clues}})

        async def _run_action(team: TeamKey, opp: TeamKey, kind: str):
            acting = red if team == "red" else blue
            # Get scratchpads for guessers
            g1_scratchpad = agent_states.get_scratchpad(f"{team}_guesser_1")
            g2_scratchpad = agent_states.get_scratchpad(f"{team}_guesser_2")
            
            action, scratchpad_adds = await run_bounded_action(
                round_inputs, team, opp, kind, 
                acting["g1"], acting["g2"],
                g1_scratchpad, g2_scratchpad
            )
            
            # Update guesser scratchpads
            if scratchpad_adds:
                for agent_id, addition in scratchpad_adds.items():
                    if addition:
                        state = agent_states.get_or_create(agent_id)
                        state.append_to_scratchpad(r, addition)
                        if show_scratchpads and not args.quiet:
                            print_scratchpad_addition(agent_id, addition, team)
            
            return action

        # Run all four actions off the same frozen snapshot (no reveal yet).
        red_decode, blue_decode, red_intercept, blue_intercept = await asyncio.gather(
            _run_action("red", "blue", "decode"),
            _run_action("blue", "red", "decode"),
            _run_action("red", "blue", "intercept"),
            _run_action("blue", "red", "intercept"),
        )

        def _print_action(a):
            if args.quiet:
                return
            label = f"{a.team.upper()} {a.kind.upper()}"
            _p(f"\n{tcol(a.team)}{Colors.BOLD}[{label}]{Colors.RESET}")
            for ind in a.independent:
                _p(f"  indep {ind.agent_id}: guess={ind.guess} conf={ind.confidence} ok={ind.parse_ok}")
                if ind.rationale:
                    _p(f"    rationale: {ind.rationale}")
            for sh in a.share:
                _p(f"  share {sh.agent_id}: {sh.message}")
            _p(f"  consensus ({a.consensus.captain_id}): guess={a.consensus.guess} conf={a.consensus.confidence} ok={a.consensus.parse_ok}")
            if a.consensus.rationale:
                _p(f"    rationale: {a.consensus.rationale}")

        _print_action(red_decode)
        _print_action(blue_decode)
        _print_action(red_intercept)
        _print_action(blue_intercept)

        # Single reveal + update (no mid-round leakage)
        true_red = base_inputs.current_codes["red"]
        true_blue = base_inputs.current_codes["blue"]

        red_decode_correct = (red_decode.consensus.guess == true_red)
        blue_decode_correct = (blue_decode.consensus.guess == true_blue)
        red_intercept_correct = (red_intercept.consensus.guess == true_blue)
        blue_intercept_correct = (blue_intercept.consensus.guess == true_red)

        counters_after = update_counters_after_round(
            counters,
            red_intercept_correct=bool(red_intercept_correct),
            blue_intercept_correct=bool(blue_intercept_correct),
            red_decode_correct=bool(red_decode_correct),
            blue_decode_correct=bool(blue_decode_correct),
        )

        # Annotate correctness into logs post-reveal
        red_decode = red_decode.model_copy(update={"correct": bool(red_decode_correct)})
        blue_decode = blue_decode.model_copy(update={"correct": bool(blue_decode_correct)})
        red_intercept = red_intercept.model_copy(update={"correct": bool(red_intercept_correct)})
        blue_intercept = blue_intercept.model_copy(update={"correct": bool(blue_intercept_correct)})

        from src.decrypto.models import RoundLog
        from src.decrypto.orchestrator import _state_tag  # reuse internal helper

        round_log = RoundLog(
            round_number=r,
            counters_before=counters,
            counters_after=counters_after,
            round_state_at_clue_time={"red": _state_tag(counters["red"]), "blue": _state_tag(counters["blue"])},
            public_clues={"red": red_clues, "blue": blue_clues},
            reveal_true_codes={"red": true_red, "blue": true_blue},
            actions=(red_decode, blue_decode, red_intercept, blue_intercept),
            private={"red": red_priv, "blue": blue_priv},
        )
        round_logs.append(round_log)
        counters = counters_after

        if not args.quiet:
            _p(f"\n{Colors.BOLD}[REVEAL]{Colors.RESET}")
            _p(f"{Colors.RED}  True RED code:{Colors.RESET}  {true_red} | decoded_ok={red_decode_correct} | intercepted_by_BLUE={blue_intercept_correct}")
            _p(f"{Colors.BLUE}  True BLUE code:{Colors.RESET} {true_blue} | decoded_ok={blue_decode_correct} | intercepted_by_RED={red_intercept_correct}")
            _p(f"{Colors.BOLD}[COUNTERS]{Colors.RESET}")
            _p(f"{Colors.RED}  RED:{Colors.RESET} {counters['red'].model_dump(mode='json')}")
            _p(f"{Colors.BLUE}  BLUE:{Colors.RESET} {counters['blue'].model_dump(mode='json')}")

        w, reason = check_winner(counters, round_number=r, max_rounds=cfg.max_rounds)
        if w is not None or reason in ("survived", "max_rounds"):
            winner = w
            result_reason = reason
            break
    else:
        winner = None
        result_reason = "max_rounds"

    # Collect final scratchpad contents
    agent_scratchpads = {
        agent_id: agent_state.scratchpad
        for agent_id, agent_state in agent_states.get_all_states().items()
        if agent_state.scratchpad
    }

    episode = DecryptoEpisodeRecord(
        episode_id=f"decrypto-{actual_seed}-{game_id}",
        timestamp=datetime.utcnow(),
        config=cfg,
        game_id=game_id,
        seed=actual_seed,
        keys=keys,
        code_sequences=code_sequences,
        rounds=tuple(round_logs),
        winner=winner,
        result_reason=result_reason,  # type: ignore[arg-type]
        scores={},
        agent_scratchpads=agent_scratchpads,
    )
    episode = episode.model_copy(update={"scores": compute_episode_scores(episode)})

    _p(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    if episode.winner:
        _p(f"{tcol(episode.winner)}{Colors.BOLD}WINNER: {episode.winner.upper()} ({episode.result_reason}){Colors.RESET}")
    else:
        _p(f"{Colors.YELLOW}{Colors.BOLD}DRAW ({episode.result_reason}){Colors.RESET}")
    _p(f"Rounds played: {len(episode.rounds)}")
    _p(f"Scores: {episode.scores}")
    _p(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    
    # Print all scratchpads at end
    if show_scratchpads and not args.quiet:
        print_all_scratchpads(agent_states)


if __name__ == "__main__":
    asyncio.run(main())
