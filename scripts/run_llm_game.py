#!/usr/bin/env python3
"""Run a single game with LLM agents with verbose output."""

import asyncio
import argparse
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

from src.engine import GameConfig, Team, GameMode, Phase, create_game, CardType
from src.agents import AgentConfig, CluerAgent, GuesserAgent, create_provider
from src.runner import TeamAgents, ExtendedEpisodeRecord
from src.runner.orchestrator import run_clue_phase, run_discussion_phase, run_guess_phase, TurnTraces
from src.metrics import compute_episode_metrics, export_metrics


# ANSI colors for terminal output
class Colors:
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
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


def create_llm_team(
    team: Team,
    cluer_model: str = "anthropic/claude-3.5-sonnet",
    guesser_1_model: str | None = None,
    guesser_2_model: str | None = None,
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
            create_provider("openrouter", cluer_model),
        ),
        guesser_1=GuesserAgent(
            AgentConfig(model=guesser_1_model, role="guesser", team=team, agent_id=f"{team_key}_guesser_1", temperature=0.7),
            create_provider("openrouter", guesser_1_model),
        ),
        guesser_2=GuesserAgent(
            AgentConfig(model=guesser_2_model, role="guesser", team=team, agent_id=f"{team_key}_guesser_2", temperature=0.7),
            create_provider("openrouter", guesser_2_model),
        ),
    )


async def run_verbose_episode(
    config: GameConfig,
    red_team: TeamAgents,
    blue_team: TeamAgents,
    max_turns: int = 50,
    max_discussion_rounds: int = 3,
    show_board: bool = True,
) -> ExtendedEpisodeRecord:
    """Run an episode with verbose output."""
    import uuid
    from datetime import datetime
    from src.engine import create_game, GameMode, DiscussionMessage
    from src.runner.orchestrator import transition_to_guessing

    episode_id = str(uuid.uuid4())[:8]
    start_time = datetime.utcnow()

    state = create_game(config=config)
    all_traces = []

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

        # Clue phase
        state, clue_trace, should_continue = await run_clue_phase(team_agents.cluer, state)

        if not should_continue:
            print(f"  {team.value} passed their turn.")
            traces = TurnTraces(
                turn_number=turn_count,
                team=team,
                clue_trace=clue_trace,
                discussion_traces=[],
                guess_trace=None,
            )
            all_traces.append(traces)
            continue

        # Print the clue
        if state.current_clue:
            print_clue(team, state.current_clue.word, state.current_clue.number)

        # Discussion phase
        guessers = team_agents.get_guessers()

        if not skip_discussion and len(guessers) >= 2:
            print(f"{team_color(team)}{Colors.BOLD}[DISCUSSION]{Colors.RESET}")

            from src.agents import run_discussion
            messages, discussion_traces, state = await run_discussion(
                guessers, state, max_discussion_rounds
            )

            # Print discussion messages
            for msg in messages:
                print_discussion(msg.agent_id, msg.content, team)

            from src.engine import transition_to_guessing
            state = transition_to_guessing(state)
        else:
            discussion_traces = []
            from src.engine import transition_to_guessing
            state = transition_to_guessing(state)

        # Guess phase
        print(f"\n{team_color(team)}{Colors.BOLD}[GUESSING]{Colors.RESET}")

        turn_number = state.turn_number
        discussion_messages = [
            e for e in state.public_transcript
            if isinstance(e, DiscussionMessage) and e.turn_number == turn_number
        ]

        guesses, guess_trace = await team_agents.primary_guesser.make_guesses(state, discussion_messages)

        from src.engine import process_guess, process_pass

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
        metadata={
            "red_team": {"model": red_team.cluer.config.model},
            "blue_team": {"model": blue_team.cluer.config.model},
        },
    )

    return episode


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

    # Red team arguments
    parser.add_argument("--red", default="anthropic/claude-3.5-sonnet",
                        help="Default model for entire red team")
    parser.add_argument("--red-cluer", default=None,
                        help="Model for red cluer (overrides --red)")
    parser.add_argument("--red-guesser-1", default=None,
                        help="Model for red guesser 1 (overrides --red)")
    parser.add_argument("--red-guesser-2", default=None,
                        help="Model for red guesser 2 (defaults to --red-guesser-1 or --red)")

    # Blue team arguments
    parser.add_argument("--blue", default="openai/gpt-4o",
                        help="Default model for entire blue team")
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
    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        "standard": GameMode.STANDARD,
        "no_assassin": GameMode.NO_ASSASSIN,
        "single_guesser": GameMode.SINGLE_GUESSER,
    }

    config = GameConfig.for_mode(mode_map[args.mode], seed=args.seed)

    # Resolve red team models
    red_cluer = args.red_cluer or args.red
    red_guesser_1 = args.red_guesser_1 or args.red
    red_guesser_2 = args.red_guesser_2 or red_guesser_1

    # Resolve blue team models
    blue_cluer = args.blue_cluer or args.blue
    blue_guesser_1 = args.blue_guesser_1 or args.blue
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

    episode = await run_verbose_episode(
        config=config,
        red_team=create_llm_team(Team.RED, red_cluer, red_guesser_1, red_guesser_2),
        blue_team=create_llm_team(Team.BLUE, blue_cluer, blue_guesser_1, blue_guesser_2),
        max_turns=args.max_turns,
        show_board=not args.quiet,
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
