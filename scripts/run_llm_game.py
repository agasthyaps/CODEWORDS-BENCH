#!/usr/bin/env python3
"""Run a single game with LLM agents with verbose output."""

import asyncio
import argparse
import random
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

from src.engine import GameConfig, Team, GameMode, Phase, create_game, CardType
from src.agents import AgentConfig, CluerAgent, GuesserAgent, create_provider
from src.runner import TeamAgents, ExtendedEpisodeRecord
from src.runner.orchestrator import TurnTraces
from src.core.state import AgentStateManager
from src.metrics import compute_episode_metrics, export_metrics
from src.benchmark import load_model_farm


# ANSI colors for terminal output
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


def team_color(team: Team) -> str:
    return Colors.RED if team == Team.RED else Colors.BLUE


def print_board(state, revealed_only=False):
    """Print the game board."""
    words = state.board.words
    key = state.board.key_by_word
    revealed = state.revealed

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}BOARD{Colors.RESET}")
    print(f"{'=' * 60}")

    for i in range(0, 25, 5):
        row_words = words[i:i+5]
        row_display = []
        for word in row_words:
            if word in revealed:
                # Revealed - show with color
                card_type = revealed[word]
                if card_type == CardType.RED:
                    row_display.append(f"{Colors.RED}{word:^12}{Colors.RESET}")
                elif card_type == CardType.BLUE:
                    row_display.append(f"{Colors.BLUE}{word:^12}{Colors.RESET}")
                elif card_type == CardType.ASSASSIN:
                    row_display.append(f"{Colors.BOLD}☠ {word:^10}{Colors.RESET}")
                else:
                    row_display.append(f"{Colors.GRAY}{word:^12}{Colors.RESET}")
            else:
                # Unrevealed
                row_display.append(f"{word:^12}")
        print(" | ".join(row_display))

    # Show remaining counts
    red_remaining = len(state.board.key_by_category["red"]) - sum(1 for w, c in revealed.items() if c == CardType.RED)
    blue_remaining = len(state.board.key_by_category["blue"]) - sum(1 for w, c in revealed.items() if c == CardType.BLUE)
    print(f"\n{Colors.RED}Red remaining: {red_remaining}{Colors.RESET} | {Colors.BLUE}Blue remaining: {blue_remaining}{Colors.RESET}")
    print(f"{'=' * 60}\n")


def print_clue(team: Team, clue_word: str, clue_number: int):
    """Print a clue."""
    color = team_color(team)
    print(f"\n{color}{Colors.BOLD}[{team.value} CLUER]{Colors.RESET}")
    print(f"{color}  Clue: {clue_word} ({clue_number}){Colors.RESET}\n")


def print_discussion(agent_id: str, message: str, team: Team):
    """Print a discussion message."""
    color = team_color(team)
    # Truncate long messages
    if len(message) > 200:
        message = message[:200] + "..."
    print(f"{color}  [{agent_id}]: {message}{Colors.RESET}")


def print_guess(team: Team, word: str, result: CardType, correct: bool):
    """Print a guess result."""
    color = team_color(team)
    if result == CardType.ASSASSIN:
        result_str = f"{Colors.BOLD}☠ ASSASSIN!{Colors.RESET}"
    elif correct:
        result_str = f"{Colors.GREEN}✓ Correct!{Colors.RESET}"
    elif result == CardType.NEUTRAL:
        result_str = f"{Colors.GRAY}○ Neutral{Colors.RESET}"
    else:
        result_str = f"{Colors.YELLOW}✗ Wrong team!{Colors.RESET}"

    print(f"{color}  Guess: {word} → {result_str}")


def print_turn_header(turn_number: int, team: Team):
    """Print turn header."""
    color = team_color(team)
    print(f"\n{color}{Colors.BOLD}{'─' * 60}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}TURN {turn_number} - {team.value}'s TURN{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{'─' * 60}{Colors.RESET}")


def print_scratchpad_addition(agent_id: str, addition: str, team: Team):
    """Print when an agent adds to their scratchpad."""
    color = team_color(team)
    # Truncate if too long
    if len(addition) > 150:
        addition = addition[:150] + "..."
    print(f"{Colors.CYAN}  📝 [{agent_id} scratchpad]: {addition}{Colors.RESET}")


def print_scratchpads(agent_states: AgentStateManager, red_team: TeamAgents, blue_team: TeamAgents):
    """Print all agent scratchpads at end of game."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}AGENT SCRATCHPADS{Colors.RESET}")
    print(f"{'=' * 60}")
    
    all_states = agent_states.get_all_states()
    
    if not any(s.scratchpad for s in all_states.values()):
        print(f"{Colors.DIM}  (no agents used scratchpads){Colors.RESET}")
        return
    
    for team, team_agents in [("RED", red_team), ("BLUE", blue_team)]:
        color = Colors.RED if team == "RED" else Colors.BLUE
        print(f"\n{color}{Colors.BOLD}{team} TEAM:{Colors.RESET}")
        
        for agent in [team_agents.cluer, team_agents.guesser_1, team_agents.guesser_2]:
            if agent is None:
                continue
            agent_id = agent.config.agent_id
            state = all_states.get(agent_id)
            if state and state.scratchpad:
                print(f"{color}  [{agent_id}]:{Colors.RESET}")
                for line in state.scratchpad.split("\n"):
                    print(f"{Colors.CYAN}    {line}{Colors.RESET}")
            else:
                print(f"{color}  [{agent_id}]: {Colors.DIM}(empty){Colors.RESET}")


def create_llm_team(
    team: Team,
    cluer_model: str,
    guesser_1_model: str | None = None,
    guesser_2_model: str | None = None,
    base_url_by_model: dict[str, str | None] | None = None,
):
    """Create a team with LLM agents.

    Args:
        team: Which team (RED or BLUE)
        cluer_model: Model for the cluer (also default for guessers if not specified)
        guesser_1_model: Model for guesser 1 (defaults to cluer_model)
        guesser_2_model: Model for guesser 2 (defaults to guesser_1_model)
    """
    team_key = team.value.lower()

    # Default guessers to cluer model if not specified
    if guesser_1_model is None:
        guesser_1_model = cluer_model
    if guesser_2_model is None:
        guesser_2_model = guesser_1_model

    return TeamAgents(
        cluer=CluerAgent(
            AgentConfig(model=cluer_model, role="cluer", team=team, agent_id=f"{team_key}_cluer", temperature=0.7),
            create_provider("openrouter", cluer_model, base_url=(base_url_by_model or {}).get(cluer_model)),
        ),
        guesser_1=GuesserAgent(
            AgentConfig(model=guesser_1_model, role="guesser", team=team, agent_id=f"{team_key}_guesser_1", temperature=0.7),
            create_provider("openrouter", guesser_1_model, base_url=(base_url_by_model or {}).get(guesser_1_model)),
        ),
        guesser_2=GuesserAgent(
            AgentConfig(model=guesser_2_model, role="guesser", team=team, agent_id=f"{team_key}_guesser_2", temperature=0.7),
            create_provider("openrouter", guesser_2_model, base_url=(base_url_by_model or {}).get(guesser_2_model)),
        ),
    )


async def run_verbose_episode(
    config: GameConfig,
    red_team: TeamAgents,
    blue_team: TeamAgents,
    max_turns: int = 50,
    max_discussion_rounds: int = 3,
    show_board: bool = True,
    show_scratchpads: bool = True,
) -> tuple[ExtendedEpisodeRecord, AgentStateManager]:
    """Run an episode with verbose output."""
    import uuid
    from datetime import datetime
    from src.engine import create_game, GameMode, DiscussionMessage, transition_to_guessing, process_guess, process_pass

    episode_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()

    state = create_game(config=config)
    all_traces = []
    
    # Initialize agent state manager for scratchpads
    agent_states = AgentStateManager()

    skip_discussion = config.mode == GameMode.SINGLE_GUESSER

    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}CODENAMES GAME - Episode {episode_id}{Colors.RESET}")
    print(f"Red Model: {red_team.cluer.config.model}")
    print(f"Blue Model: {blue_team.cluer.config.model}")
    print(f"Seed: {config.seed}")
    print(f"{'=' * 60}")

    if show_board:
        print_board(state)

    turn_count = 0
    while state.winner is None and turn_count < max_turns:
        turn_count += 1
        team = state.current_turn
        team_agents = red_team if team == Team.RED else blue_team

        print_turn_header(turn_count, team)

        # Clue phase with scratchpad
        cluer = team_agents.cluer
        cluer_scratchpad = agent_states.get_scratchpad(cluer.config.agent_id)
        
        clue, clue_trace, clue_scratchpad_add = await cluer.generate_clue(state, cluer_scratchpad)
        
        # Update scratchpad if agent added to it
        if clue_scratchpad_add:
            agent_state = agent_states.get_or_create(cluer.config.agent_id)
            agent_state.append_to_scratchpad(turn_count, clue_scratchpad_add)
            if show_scratchpads:
                print_scratchpad_addition(cluer.config.agent_id, clue_scratchpad_add, team)

        if clue is None:
            print(f"  {team.value} passed their turn.")
            from src.engine import end_turn
            state = end_turn(state)
            traces = TurnTraces(
                turn_number=turn_count,
                team=team,
                clue_trace=clue_trace,
                discussion_traces=[],
                guess_trace=None,
            )
            all_traces.append(traces)
            continue

        # Apply clue to state
        from src.engine import apply_clue
        state = apply_clue(state, clue.word, clue.number)

        # Print the clue
        print_clue(team, clue.word, clue.number)

        # Discussion phase
        guessers = team_agents.get_guessers()
        discussion_traces = []

        if not skip_discussion and len(guessers) >= 2:
            print(f"{team_color(team)}{Colors.BOLD}[DISCUSSION]{Colors.RESET}")

            # Run discussion manually to capture scratchpads
            from src.agents.guesser import parse_discussion_response
            messages = []
            consecutive_consensus = 0
            previous_top_words = None
            max_messages = max_discussion_rounds * 2

            for i in range(max_messages):
                guesser = guessers[i % 2]
                guesser_scratchpad = agent_states.get_scratchpad(guesser.config.agent_id)
                
                message, trace, scratchpad_add = await guesser.discuss(state, messages, guesser_scratchpad)
                discussion_traces.append(trace)
                
                # Update scratchpad
                if scratchpad_add:
                    agent_state = agent_states.get_or_create(guesser.config.agent_id)
                    agent_state.append_to_scratchpad(turn_count, scratchpad_add)
                    if show_scratchpads:
                        print_scratchpad_addition(guesser.config.agent_id, scratchpad_add, team)

                # Add to state and messages
                from src.engine import add_discussion_message
                state = add_discussion_message(state, message.agent_id, message.content)
                actual_message = state.public_transcript[-1]
                if isinstance(actual_message, DiscussionMessage):
                    messages.append(actual_message)
                else:
                    messages.append(message)

                print_discussion(message.agent_id, message.content, team)

                # Check for consensus
                parsed = parse_discussion_response(message.content)
                if parsed.consensus:
                    if consecutive_consensus == 0:
                        consecutive_consensus = 1
                        previous_top_words = parsed.top_words
                    else:
                        # Check if TOP lists match
                        set1 = set(w.upper() for w in (previous_top_words or []))
                        set2 = set(w.upper() for w in (parsed.top_words or []))
                        if set1 == set2:
                            consecutive_consensus = 2
                            break
                        else:
                            consecutive_consensus = 1
                            previous_top_words = parsed.top_words
                else:
                    consecutive_consensus = 0
                    previous_top_words = None

            state = transition_to_guessing(state)
        else:
            state = transition_to_guessing(state)

        # Guess phase
        print(f"\n{team_color(team)}{Colors.BOLD}[GUESSING]{Colors.RESET}")

        turn_number = state.turn_number
        discussion_messages = [
            e for e in state.public_transcript
            if isinstance(e, DiscussionMessage) and e.turn_number == turn_number
        ]

        guesser = team_agents.primary_guesser
        guesser_scratchpad = agent_states.get_scratchpad(guesser.config.agent_id)
        
        guesses, guess_trace, guess_scratchpad_add = await guesser.make_guesses(
            state, discussion_messages, guesser_scratchpad
        )
        
        # Update scratchpad
        if guess_scratchpad_add:
            agent_state = agent_states.get_or_create(guesser.config.agent_id)
            agent_state.append_to_scratchpad(turn_count, guess_scratchpad_add)
            if show_scratchpads:
                print_scratchpad_addition(guesser.config.agent_id, guess_scratchpad_add, team)

        if not guesses:
            print(f"  {team.value} passed without guessing.")
            state = process_pass(state)
        else:
            for guess_word in guesses:
                state, result, turn_continues = process_guess(guess_word, state)

                # Find the guess event we just added
                last_guess = next(
                    (e for e in reversed(state.public_transcript)
                     if hasattr(e, 'event_type') and e.event_type == "guess"),
                    None
                )
                if last_guess:
                    print_guess(team, guess_word, CardType(last_guess.result), last_guess.correct)

                if state.winner is not None:
                    break
                if not turn_continues:
                    break

            # End turn if still in guess phase
            if state.phase == Phase.GUESS and state.winner is None:
                state = process_pass(state)

        traces = TurnTraces(
            turn_number=turn_count,
            team=team,
            clue_trace=clue_trace,
            discussion_traces=discussion_traces,
            guess_trace=guess_trace,
        )
        all_traces.append(traces)

        if show_board and state.winner is None:
            print_board(state)

        if state.phase == Phase.GAME_OVER:
            break

    # Print final result
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    if state.winner:
        winner_color = team_color(state.winner)
        print(f"{winner_color}{Colors.BOLD}🎉 {state.winner.value} WINS! 🎉{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}Game ended in a draw (max turns reached){Colors.RESET}")
    print(f"Total turns: {turn_count}")
    print(f"{'=' * 60}\n")

    if show_board:
        print_board(state)
    
    # Print all scratchpads at end
    if show_scratchpads:
        print_scratchpads(agent_states, red_team, blue_team)

    # Collect final scratchpad contents
    agent_scratchpads = {
        agent_id: agent_state.scratchpad
        for agent_id, agent_state in agent_states.get_all_states().items()
        if agent_state.scratchpad
    }

    from src.engine import Board
    episode = ExtendedEpisodeRecord(
        episode_id=episode_id,
        timestamp=start_time,
        config=config,
        board_seed=state.board_seed,
        board=state.board,
        public_transcript=[e.model_dump() for e in state.public_transcript],
        turn_traces=all_traces,
        winner=state.winner,
        total_turns=turn_count,
        agent_scratchpads=agent_scratchpads,
        metadata={
            "red_team": {"model": red_team.cluer.config.model},
            "blue_team": {"model": blue_team.cluer.config.model},
        },
    )

    return episode, agent_states


async def main():
    parser = argparse.ArgumentParser(
        description="Run a Codenames game with LLM agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Same model for both teams
  python run_llm_game.py --red anthropic/claude-3.5-sonnet --blue openai/gpt-4o

  # Different cluer and guessers for red team
  python run_llm_game.py --red openai/gpt-4o --red-cluer anthropic/claude-3.5-sonnet

  # Fully custom team composition
  python run_llm_game.py --red-cluer anthropic/claude-3.5-sonnet \\
                         --red-guesser-1 openai/gpt-4o \\
                         --red-guesser-2 openai/gpt-4o-mini \\
                         --blue openai/gpt-4o
        """
    )

    parser.add_argument(
        "--random-teams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly choose heterogeneous role assignments from the model farm (default: on).",
    )
    parser.add_argument(
        "--models-config",
        default=None,
        help="Path to model farm JSON (default: config/models.json if present).",
    )
    parser.add_argument(
        "--allow-cross-team-duplicates",
        action="store_true",
        help="Allow the same model to appear on both teams (still enforces heterogeneity within each team).",
    )

    # Red team arguments
    parser.add_argument("--red", default=None, help="Default model for entire red team")
    parser.add_argument("--red-cluer", default=None,
                        help="Model for red cluer (overrides --red)")
    parser.add_argument("--red-guesser-1", default=None,
                        help="Model for red guesser 1 (overrides --red)")
    parser.add_argument("--red-guesser-2", default=None,
                        help="Model for red guesser 2 (defaults to --red-guesser-1 or --red)")

    # Blue team arguments
    parser.add_argument("--blue", default=None, help="Default model for entire blue team")
    parser.add_argument("--blue-cluer", default=None,
                        help="Model for blue cluer (overrides --blue)")
    parser.add_argument("--blue-guesser-1", default=None,
                        help="Model for blue guesser 1 (overrides --blue)")
    parser.add_argument("--blue-guesser-2", default=None,
                        help="Model for blue guesser 2 (defaults to --blue-guesser-1 or --blue)")

    # Game settings
    parser.add_argument("--mode", choices=["standard", "no_assassin", "single_guesser"],
                        default="standard", help="Game mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for board generation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output (no board display)")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns before draw")
    parser.add_argument("--no-scratchpads", action="store_true", help="Hide scratchpad output")
    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        "standard": GameMode.STANDARD,
        "no_assassin": GameMode.NO_ASSASSIN,
        "single_guesser": GameMode.SINGLE_GUESSER,
    }

    config = GameConfig.for_mode(mode_map[args.mode], seed=args.seed)

    # Load model farm (for random teams and base_url mapping)
    repo_root = Path(__file__).parent.parent
    default_models_config_path = repo_root / "config" / "models.json"
    models_config_path = Path(args.models_config) if args.models_config else None
    load_path = None
    if models_config_path and models_config_path.exists():
        load_path = models_config_path
    elif default_models_config_path.exists():
        load_path = default_models_config_path

    model_ids: list[str] = []
    base_url_by_model: dict[str, str | None] = {}
    if load_path is not None:
        farm_models, _ = load_model_farm(load_path)
        # Deduplicate model IDs while preserving order
        seen = set()
        for m in farm_models:
            if m.model_id in seen:
                continue
            seen.add(m.model_id)
            model_ids.append(m.model_id)
            base_url_by_model[m.model_id] = m.base_url

    def _sample_distinct(pool: list[str], k: int, rng: random.Random) -> list[str]:
        if len(pool) < k:
            raise ValueError(f"Need at least {k} distinct models, only have {len(pool)} in model farm.")
        return rng.sample(pool, k)

    def _pick_heterogeneous_teams(rng: random.Random) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
        if len(model_ids) < 3:
            raise ValueError(
                "Random team selection requires at least 3 models in config/models.json "
                "(to make one heterogeneous team)."
            )
        if not args.allow_cross_team_duplicates and len(model_ids) >= 6:
            picks = _sample_distinct(model_ids, 6, rng)
            red = (picks[0], picks[1], picks[2])
            blue = (picks[3], picks[4], picks[5])
            return red, blue

        # Allow cross-team duplicates (but keep each team heterogeneous)
        red = tuple(_sample_distinct(model_ids, 3, rng))  # type: ignore[assignment]
        blue = tuple(_sample_distinct(model_ids, 3, rng))  # type: ignore[assignment]
        return red, blue

    any_team_override = any(
        [
            args.red, args.blue,
            args.red_cluer, args.red_guesser_1, args.red_guesser_2,
            args.blue_cluer, args.blue_guesser_1, args.blue_guesser_2,
        ]
    )

    if args.random_teams and not any_team_override:
        rng = random.Random(args.seed)
        (red_cluer, red_guesser_1, red_guesser_2), (blue_cluer, blue_guesser_1, blue_guesser_2) = _pick_heterogeneous_teams(rng)
    else:
        # Resolve via CLI (fallback to defaults if user specifies partial)
        default_red = args.red or (model_ids[0] if model_ids else "openai/gpt-4o")
        default_blue = args.blue or (model_ids[1] if len(model_ids) > 1 else default_red)

        red_cluer = args.red_cluer or default_red
        red_guesser_1 = args.red_guesser_1 or default_red
        red_guesser_2 = args.red_guesser_2 or red_guesser_1

        blue_cluer = args.blue_cluer or default_blue
        blue_guesser_1 = args.blue_guesser_1 or default_blue
        blue_guesser_2 = args.blue_guesser_2 or blue_guesser_1

    # Print team composition
    print(f"\n{Colors.BOLD}Team Composition:{Colors.RESET}")
    print(f"{Colors.RED}Red Team:{Colors.RESET}")
    print(f"  Cluer:    {red_cluer}")
    print(f"  Guesser1: {red_guesser_1}")
    print(f"  Guesser2: {red_guesser_2}")
    print(f"{Colors.BLUE}Blue Team:{Colors.RESET}")
    print(f"  Cluer:    {blue_cluer}")
    print(f"  Guesser1: {blue_guesser_1}")
    print(f"  Guesser2: {blue_guesser_2}")
    print(f"\nMode: {args.mode} | Seed: {args.seed}")

    red_team = create_llm_team(Team.RED, red_cluer, red_guesser_1, red_guesser_2, base_url_by_model=base_url_by_model)
    blue_team = create_llm_team(Team.BLUE, blue_cluer, blue_guesser_1, blue_guesser_2, base_url_by_model=base_url_by_model)

    episode, agent_states = await run_verbose_episode(
        config=config,
        red_team=red_team,
        blue_team=blue_team,
        max_turns=args.max_turns,
        show_board=not args.quiet,
        show_scratchpads=not args.no_scratchpads,
    )

    # Print metrics summary
    metrics = compute_episode_metrics(episode)
    print(f"\n{Colors.BOLD}METRICS SUMMARY{Colors.RESET}")
    print(f"Red Coordination Score: {metrics.red_coordination_score:.3f}")
    print(f"Blue Coordination Score: {metrics.blue_coordination_score:.3f}")
    print(f"Red Guess Accuracy: {metrics.red_metrics.guess_accuracy:.1%}")
    print(f"Blue Guess Accuracy: {metrics.blue_metrics.guess_accuracy:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
