"""Core game logic for Codenames."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from .models import (
    Board, CardType, Clue, GameConfig, GameState, Guess, Pass,
    Phase, Team, TranscriptEvent, DiscussionMessage
)


def load_wordlist(path: Path | None = None) -> list[str]:
    """Load the word list from file."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "wordlist.txt"

    with open(path, "r") as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return words


def generate_board(
    word_list: list[str] | None = None,
    config: GameConfig | None = None
) -> tuple[Board, int]:
    """
    Generate a game board with random word selection and key assignment.

    Returns:
        Tuple of (Board, seed_used) for reproducibility.
    """
    if config is None:
        config = GameConfig()

    if word_list is None:
        word_list = load_wordlist()

    # Determine seed
    if config.seed is not None:
        seed = config.seed
    else:
        seed = random.randint(0, 2**31 - 1)

    rng = random.Random(seed)

    # Select 25 words
    if len(word_list) < config.words_per_board:
        raise ValueError(
            f"Word list has {len(word_list)} words, need {config.words_per_board}"
        )

    words = rng.sample(word_list, config.words_per_board)

    # Assign card types
    total_needed = (
        config.red_count + config.blue_count +
        config.neutral_count + config.assassin_count
    )
    if total_needed != config.words_per_board:
        raise ValueError(
            f"Card counts ({total_needed}) don't match words_per_board ({config.words_per_board})"
        )

    # Create shuffled assignment
    assignments = (
        [CardType.RED] * config.red_count +
        [CardType.BLUE] * config.blue_count +
        [CardType.NEUTRAL] * config.neutral_count +
        [CardType.ASSASSIN] * config.assassin_count
    )
    rng.shuffle(assignments)

    # Build key structures
    key_by_word: dict[str, CardType] = {}
    key_by_category: dict[str, set[str]] = {
        "red": set(),
        "blue": set(),
        "neutral": set(),
        "assassin": set(),
    }

    for word, card_type in zip(words, assignments):
        key_by_word[word] = card_type
        key_by_category[card_type.value.lower()].add(word)

    board = Board(
        words=words,
        key_by_category=key_by_category,
        key_by_word=key_by_word,
    )

    return board, seed


def create_game(
    word_list: list[str] | None = None,
    config: GameConfig | None = None
) -> GameState:
    """Create a new game with the given configuration."""
    if config is None:
        config = GameConfig()

    board, seed = generate_board(word_list, config)

    return GameState(
        config=config,
        board=board,
        board_seed=seed,
        current_turn=config.starting_team,
        phase=Phase.CLUE,
    )


def validate_clue(
    word: str,
    number: int,
    state: GameState
) -> tuple[bool, str | None]:
    """
    Validate a clue against game rules.

    Returns:
        (is_valid, error_message) - error_message is None if valid.
    """
    word = word.upper().strip()

    # Check for non-alpha characters (allow only letters)
    if not word.isalpha():
        return False, f"Clue '{word}' contains non-alphabetic characters"

    # Check for empty clue
    if not word:
        return False, "Clue cannot be empty"

    # Check number range
    if number < -1:
        return False, f"Number {number} is invalid (minimum is -1 for unlimited)"

    if number == -1 and not state.config.allow_unlimited_clue:
        return False, "Unlimited clues are not allowed in this game"

    if number > state.config.max_clue_number:
        return False, f"Number {number} exceeds maximum ({state.config.max_clue_number})"

    # Check for exact match with board word (case-insensitive)
    board_words_upper = [w.upper() for w in state.board.words]
    if word in board_words_upper:
        return False, f"Clue '{word}' is a word on the board"

    # Check for substring relationships (both directions)
    for board_word in board_words_upper:
        if word in board_word:
            return False, f"Clue '{word}' is a substring of board word '{board_word}'"
        if board_word in word:
            return False, f"Board word '{board_word}' is a substring of clue '{word}'"

    # Check for previously used clues
    for event in state.public_transcript:
        if isinstance(event, Clue) and event.word.upper() == word:
            return False, f"Clue '{word}' was already used"

    return True, None


def add_event(state: GameState, event: TranscriptEvent) -> GameState:
    """Add an event to the transcript with proper indexing."""
    new_state = state.model_copy(deep=True)
    new_state.event_counter += 1

    # Update event's index
    if isinstance(event, Clue):
        event = Clue(
            turn_number=event.turn_number,
            event_index=new_state.event_counter,
            team=event.team,
            word=event.word,
            number=event.number,
        )
    elif isinstance(event, Guess):
        event = Guess(
            turn_number=event.turn_number,
            event_index=new_state.event_counter,
            team=event.team,
            word=event.word,
            result=event.result,
            correct=event.correct,
        )
    elif isinstance(event, Pass):
        event = Pass(
            turn_number=event.turn_number,
            event_index=new_state.event_counter,
            team=event.team,
        )
    elif isinstance(event, DiscussionMessage):
        event = DiscussionMessage(
            turn_number=event.turn_number,
            event_index=new_state.event_counter,
            team=event.team,
            agent_id=event.agent_id,
            content=event.content,
        )

    new_state.public_transcript.append(event)
    return new_state


def apply_clue(state: GameState, word: str, number: int) -> GameState:
    """Apply a valid clue to the game state."""
    if state.phase != Phase.CLUE:
        raise ValueError(f"Cannot give clue in phase {state.phase}")

    if state.winner is not None:
        raise ValueError("Game is already over")

    is_valid, error = validate_clue(word, number, state)
    if not is_valid:
        raise ValueError(f"Invalid clue: {error}")

    word = word.upper().strip()

    clue = Clue(
        turn_number=state.turn_number,
        event_index=0,  # Will be set by add_event
        team=state.current_turn,
        word=word,
        number=number,
    )

    new_state = add_event(state, clue)
    new_state.current_clue = new_state.public_transcript[-1]  # Get the clue with proper index

    # Calculate guesses remaining
    if number == -1:  # Unlimited
        new_state.guesses_remaining = 25  # Can guess all remaining
    elif number == 0:
        new_state.guesses_remaining = 1  # Zero clue allows 1 guess
    else:
        new_state.guesses_remaining = number + 1  # Standard: N + 1 guesses

    new_state.phase = Phase.DISCUSSION

    return new_state


def end_turn(state: GameState) -> GameState:
    """End the current turn and switch to the other team."""
    new_state = state.model_copy(deep=True)

    # Switch teams
    new_state.current_turn = Team.BLUE if state.current_turn == Team.RED else Team.RED
    new_state.turn_number += 1
    new_state.phase = Phase.CLUE
    new_state.current_clue = None
    new_state.guesses_remaining = 0

    return new_state


def check_winner(state: GameState) -> Team | None:
    """
    Check if there's a winner.

    Returns:
        Winning team or None if game continues.
    """
    # Check if assassin was revealed
    assassin_words = state.board.key_by_category.get("assassin", set())
    for word in assassin_words:
        if word in state.revealed:
            # The team that revealed it loses (other team wins)
            # We need to look at who made the guess
            for event in reversed(state.public_transcript):
                if isinstance(event, Guess) and event.word == word:
                    return Team.BLUE if event.team == Team.RED else Team.RED

    # Check if all team words are revealed
    red_words = state.board.key_by_category.get("red", set())
    blue_words = state.board.key_by_category.get("blue", set())

    red_revealed = all(word in state.revealed for word in red_words)
    blue_revealed = all(word in state.revealed for word in blue_words)

    if red_revealed:
        return Team.RED
    if blue_revealed:
        return Team.BLUE

    return None


def process_guess(
    word: str,
    state: GameState
) -> tuple[GameState, CardType | None, bool]:
    """
    Process a guess.

    Returns:
        (new_state, result, turn_continues)
        - result is None if guess was invalid
        - turn_continues indicates if the team can keep guessing
    """
    if state.phase != Phase.GUESS:
        raise ValueError(f"Cannot guess in phase {state.phase}")

    if state.winner is not None:
        raise ValueError("Game is already over")

    word = word.upper().strip()

    # Check if word is on board
    if word not in state.board.key_by_word:
        # Invalid guess - turn ends, nothing revealed
        new_state = end_turn(state)
        return new_state, None, False

    # Check if already revealed
    if word in state.revealed:
        # Invalid guess - turn ends, nothing revealed
        new_state = end_turn(state)
        return new_state, None, False

    # Valid guess - reveal the card
    card_type = state.board.key_by_word[word]
    correct = (
        (card_type == CardType.RED and state.current_turn == Team.RED) or
        (card_type == CardType.BLUE and state.current_turn == Team.BLUE)
    )

    guess = Guess(
        turn_number=state.turn_number,
        event_index=0,  # Will be set by add_event
        team=state.current_turn,
        word=word,
        result=card_type,
        correct=correct,
    )

    new_state = add_event(state, guess)
    new_state.revealed[word] = card_type
    new_state.guesses_remaining -= 1

    # Check for winner
    winner = check_winner(new_state)
    if winner is not None:
        new_state.winner = winner
        new_state.phase = Phase.GAME_OVER
        return new_state, card_type, False

    # Determine if turn continues
    if card_type == CardType.ASSASSIN:
        # Already handled in check_winner
        pass
    elif not correct:
        # Wrong card (opponent or neutral) - turn ends
        new_state = end_turn(new_state)
        return new_state, card_type, False
    elif new_state.guesses_remaining <= 0:
        # Exhausted guesses - turn ends
        new_state = end_turn(new_state)
        return new_state, card_type, False
    else:
        # Correct guess with guesses remaining - can continue
        return new_state, card_type, True


def process_pass(state: GameState) -> GameState:
    """Process a pass (voluntary end of guessing)."""
    if state.phase != Phase.GUESS:
        raise ValueError(f"Cannot pass in phase {state.phase}")

    if state.winner is not None:
        raise ValueError("Game is already over")

    pass_event = Pass(
        turn_number=state.turn_number,
        event_index=0,  # Will be set by add_event
        team=state.current_turn,
    )

    new_state = add_event(state, pass_event)
    new_state = end_turn(new_state)

    return new_state


def transition_to_guessing(state: GameState) -> GameState:
    """Transition from discussion phase to guessing phase."""
    if state.phase != Phase.DISCUSSION:
        raise ValueError(f"Cannot transition to guessing from phase {state.phase}")

    new_state = state.model_copy(deep=True)
    new_state.phase = Phase.GUESS
    return new_state


def add_discussion_message(
    state: GameState,
    agent_id: str,
    content: str
) -> GameState:
    """Add a discussion message to the transcript."""
    if state.phase != Phase.DISCUSSION:
        raise ValueError(f"Cannot add discussion in phase {state.phase}")

    message = DiscussionMessage(
        turn_number=state.turn_number,
        event_index=0,  # Will be set by add_event
        team=state.current_turn,
        agent_id=agent_id,
        content=content,
    )

    return add_event(state, message)


def get_visible_state(
    state: GameState,
    role: str
) -> dict[str, Any]:
    """
    Get the visible state for a specific role.

    Roles: "red_cluer", "blue_cluer", "red_guesser_1", "red_guesser_2",
           "blue_guesser_1", "blue_guesser_2"

    Cluers see the key, guessers do not.
    Everyone sees the full public transcript.
    """
    valid_roles = {
        "red_cluer", "blue_cluer",
        "red_guesser_1", "red_guesser_2",
        "blue_guesser_1", "blue_guesser_2",
    }

    if role not in valid_roles:
        raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

    is_cluer = role.endswith("_cluer")
    team = Team.RED if role.startswith("red") else Team.BLUE

    # Build visible state
    visible: dict[str, Any] = {
        "role": role,
        "team": team.value,
        "board_words": state.board.words.copy(),
        "revealed": dict(state.revealed),
        "current_turn": state.current_turn.value,
        "phase": state.phase.value,
        "turn_number": state.turn_number,
        "public_transcript": [
            event.model_dump() for event in state.public_transcript
        ],
        "winner": state.winner.value if state.winner else None,
    }

    # Add current clue info if in discussion or guess phase
    if state.current_clue is not None:
        visible["current_clue"] = {
            "word": state.current_clue.word,
            "number": state.current_clue.number,
            "team": state.current_clue.team.value,
        }
        visible["guesses_remaining"] = state.guesses_remaining

    # Cluers get the key
    if is_cluer:
        visible["key"] = {
            "red": sorted(list(state.board.key_by_category["red"])),
            "blue": sorted(list(state.board.key_by_category["blue"])),
            "neutral": sorted(list(state.board.key_by_category["neutral"])),
            "assassin": sorted(list(state.board.key_by_category["assassin"])),
        }
    # Note: guessers do NOT get a key field (not null - absent)

    return visible


def get_unrevealed_words(state: GameState, team: Team) -> list[str]:
    """Get unrevealed words for a team."""
    category = "red" if team == Team.RED else "blue"
    team_words = state.board.key_by_category[category]
    return [w for w in team_words if w not in state.revealed]


def get_all_unrevealed_words(state: GameState) -> list[str]:
    """Get all unrevealed words on the board."""
    return [w for w in state.board.words if w not in state.revealed]
