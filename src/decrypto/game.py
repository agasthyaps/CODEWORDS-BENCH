from __future__ import annotations

import itertools
import random
import uuid
from pathlib import Path

from .models import (
    CodeTriple,
    DecryptoConfig,
    RoundCounters,
    TeamKey,
)


def _repo_root() -> Path:
    # src/decrypto/game.py -> src/decrypto -> src -> repo
    return Path(__file__).resolve().parent.parent.parent


def load_keyword_bank(path: str | None) -> list[str]:
    if path is None:
        path = str(_repo_root() / "data" / "wordlist.txt")
    p = Path(path)
    words = []
    with open(p, "r") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            words.append(w.upper())
    if len(words) < 8:
        raise ValueError(f"Keyword bank too small: {len(words)} words")
    return words


def generate_all_codes() -> list[tuple[int, int, int]]:
    """
    All 24 possible codes: permutations of length 3 from digits 1..4.
    """
    digits = [1, 2, 3, 4]
    return list(itertools.permutations(digits, 3))


def pre_generate_code_sequence(seed: int) -> tuple[tuple[int, int, int], ...]:
    """
    Deterministic full permutation of the 24 codes, shuffled by seed.
    """
    rng = random.Random(seed)
    codes = generate_all_codes()
    rng.shuffle(codes)
    return tuple(codes)


def generate_keys(bank: list[str], seed: int) -> dict[TeamKey, tuple[str, str, str, str]]:
    """
    Deterministically select disjoint 4-word keys for both teams.
    """
    rng = random.Random(seed)
    picks = rng.sample(bank, 8)
    red = tuple(picks[:4])
    blue = tuple(picks[4:])
    return {"red": red, "blue": blue}  # type: ignore[return-value]


def initial_counters() -> dict[TeamKey, RoundCounters]:
    # Counters are defined from each team's perspective.
    red = RoundCounters(
        own_interceptions=0,
        own_miscommunications=0,
        opp_interceptions=0,
        opp_miscommunications=0,
    )
    blue = RoundCounters(
        own_interceptions=0,
        own_miscommunications=0,
        opp_interceptions=0,
        opp_miscommunications=0,
    )
    return {"red": red, "blue": blue}


def update_counters_after_round(
    counters_before: dict[TeamKey, RoundCounters],
    *,
    red_intercept_correct: bool,
    blue_intercept_correct: bool,
    red_decode_correct: bool,
    blue_decode_correct: bool,
) -> dict[TeamKey, RoundCounters]:
    """
    Update counters symmetrically based on round outcomes.
    - If RED intercepted BLUE correctly: RED own_interceptions++, BLUE opp_interceptions++.
    - If RED decoded own code incorrectly: RED own_miscommunications++, BLUE opp_miscommunications++.
    """
    r = counters_before["red"]
    b = counters_before["blue"]

    r_own_int = r.own_interceptions + (1 if red_intercept_correct else 0)
    b_opp_int = b.opp_interceptions + (1 if red_intercept_correct else 0)

    b_own_int = b.own_interceptions + (1 if blue_intercept_correct else 0)
    r_opp_int = r.opp_interceptions + (1 if blue_intercept_correct else 0)

    r_own_mis = r.own_miscommunications + (1 if not red_decode_correct else 0)
    b_opp_mis = b.opp_miscommunications + (1 if not red_decode_correct else 0)

    b_own_mis = b.own_miscommunications + (1 if not blue_decode_correct else 0)
    r_opp_mis = r.opp_miscommunications + (1 if not blue_decode_correct else 0)

    red_after = RoundCounters(
        own_interceptions=r_own_int,
        own_miscommunications=r_own_mis,
        opp_interceptions=r_opp_int,
        opp_miscommunications=r_opp_mis,
    )
    blue_after = RoundCounters(
        own_interceptions=b_own_int,
        own_miscommunications=b_own_mis,
        opp_interceptions=b_opp_int,
        opp_miscommunications=b_opp_mis,
    )
    return {"red": red_after, "blue": blue_after}


def check_winner(
    counters: dict[TeamKey, RoundCounters],
    *,
    round_number: int,
    max_rounds: int,
) -> tuple[TeamKey | None, str | None]:
    """
    Returns (winner_team, reason) where reason is one of:
    - interceptions
    - miscommunications
    - survived
    - max_rounds
    """
    r = counters["red"]
    b = counters["blue"]

    if r.own_interceptions >= 2:
        return "red", "interceptions"
    if b.own_interceptions >= 2:
        return "blue", "interceptions"

    if r.own_miscommunications >= 2:
        return "blue", "miscommunications"
    if b.own_miscommunications >= 2:
        return "red", "miscommunications"

    if round_number >= max_rounds:
        return None, "survived"

    return None, None


def create_game(config: DecryptoConfig) -> tuple[str, dict[TeamKey, tuple[str, str, str, str]], dict[TeamKey, tuple[tuple[int, int, int], ...]]]:
    """
    Create a new deterministic game: game_id, keys, and per-team code sequences.

    Note: codes are pre-generated for determinism. We use distinct seeds per team
    derived from the game seed to avoid perfectly mirrored code order.
    """
    bank = load_keyword_bank(config.keyword_bank_path)
    keys = generate_keys(bank, config.seed)

    # Derive per-team sequences deterministically.
    red_seq = pre_generate_code_sequence(config.seed ^ 0xA11CE)  # stable salt
    blue_seq = pre_generate_code_sequence(config.seed ^ 0xB10E5)

    # Validate code triples once.
    for t in (red_seq[0], blue_seq[0]):
        CodeTriple.validate_digits(t)  # type: ignore[arg-type]

    game_id = str(uuid.uuid4())[:8]
    return game_id, keys, {"red": red_seq, "blue": blue_seq}

