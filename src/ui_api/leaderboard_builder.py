"""Leaderboard builder - scans episodes and computes model rankings."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ModelStats(BaseModel):
    """Aggregated stats for a single model."""

    model: str
    # Codenames
    codenames_games: int = 0
    codenames_wins: int = 0
    # Decrypto
    decrypto_games: int = 0
    decrypto_wins: int = 0
    # Hanabi
    hanabi_games: int = 0
    hanabi_total_score: int = 0


class CodenamesRanking(BaseModel):
    """Codenames ranking entry."""

    model: str
    games: int
    wins: int
    win_rate: float


class DecryptoRanking(BaseModel):
    """Decrypto ranking entry."""

    model: str
    games: int
    wins: int
    win_rate: float


class HanabiRanking(BaseModel):
    """Hanabi ranking entry."""

    model: str
    games: int
    avg_score: float
    score_pct: float  # As percentage of max 25


class OverallRanking(BaseModel):
    """Overall model ranking."""

    rank: int
    model: str
    games_played: int
    overall_score: float  # 0-100 composite
    codenames_score: float | None = None
    decrypto_score: float | None = None
    hanabi_score: float | None = None


class LeaderboardData(BaseModel):
    """Complete leaderboard data structure."""

    generated_at: str
    total_episodes: dict[str, int]
    overall_rankings: list[OverallRanking]
    codenames_rankings: list[CodenamesRanking]
    decrypto_rankings: list[DecryptoRanking]
    hanabi_rankings: list[HanabiRanking]


def _benchmark_results_dir() -> Path:
    """Get the benchmark_results directory (persistent storage)."""
    # Reuse the same function that cloud_benchmark uses (handles env var)
    from src.cloud_benchmark.config import get_data_dir
    return get_data_dir()


def _sessions_dir() -> Path:
    """Get the sessions directory within benchmark_results."""
    return _benchmark_results_dir() / "sessions"


def _normalize_model_name(model_id: str) -> str:
    """Normalize model ID to a display name."""
    # Remove provider prefix for cleaner display
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id


def _extract_codenames_models(episode: dict) -> list[tuple[str, bool]]:
    """
    Extract models from a Codenames episode.

    Returns list of (model, won) tuples for each model that participated.
    Handles both UI sessions (cluer) and benchmark (cluer_model) formats.
    """
    metadata = episode.get("metadata", {})
    winner = (episode.get("winner") or "").upper()
    results = []

    # Red team - try both formats
    red_team = metadata.get("red_team", {})
    red_cluer = red_team.get("cluer") or red_team.get("cluer_model")
    if red_cluer:
        red_won = winner == "RED"
        results.append((red_cluer, red_won))

    # Blue team - try both formats
    blue_team = metadata.get("blue_team", {})
    blue_cluer = blue_team.get("cluer") or blue_team.get("cluer_model")
    if blue_cluer:
        blue_won = winner == "BLUE"
        results.append((blue_cluer, blue_won))

    return results


def _extract_decrypto_models(episode: dict) -> list[tuple[str, bool]]:
    """
    Extract models from a Decrypto episode.

    Returns list of (model, won) tuples.
    """
    metadata = episode.get("metadata", {})
    winner = (episode.get("winner") or "").lower()
    results = []

    # Red team
    red_team = metadata.get("red_team", {})
    red_cluer = red_team.get("cluer_model")
    if red_cluer:
        red_won = winner == "red"
        results.append((red_cluer, red_won))

    # Blue team
    blue_team = metadata.get("blue_team", {})
    blue_cluer = blue_team.get("cluer_model")
    if blue_cluer:
        blue_won = winner == "blue"
        results.append((blue_cluer, blue_won))

    return results


def _extract_hanabi_models(episode: dict) -> list[tuple[str, int]]:
    """
    Extract models from a Hanabi episode.

    Returns list of (model, score) tuples.
    Since Hanabi is cooperative with same model for all players,
    we return just one entry per unique model.
    """
    metadata = episode.get("metadata", {})
    player_models = metadata.get("player_models", {})
    score = episode.get("final_score", 0)

    # Get unique models (usually all same)
    unique_models = set(player_models.values())

    return [(model, score) for model in unique_models if model]


def _detect_game_type(episode: dict, filename: str) -> str | None:
    """Detect game type from episode data or filename."""
    # Check filename patterns first
    fname = filename.lower()
    if "hanabi" in fname:
        return "hanabi"
    if "decrypto" in fname:
        return "decrypto"

    # Check content
    if "final_score" in episode and "turns" in episode:
        return "hanabi"
    if "code_sequences" in episode or "result_reason" in episode:
        return "decrypto"
    if "board" in episode and "winner" in episode:
        return "codenames"

    return None


def _load_episode_file(path: Path) -> dict | None:
    """Load a single episode file, returning None on failure."""
    if path.name.startswith("."):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def scan_all_episodes() -> dict[str, list[dict]]:
    """
    Scan all episode files from benchmark_results/ (consolidated storage).

    Scans multiple patterns:
    - benchmark_results/sessions/{codenames,decrypto,hanabi}/ (UI sessions)
    - benchmark_results/*/episodes/ (cloud benchmark - flat episodes dir)
    - benchmark_results/*/{codenames,decrypto,hanabi}/ (cloud benchmark - by game type)
    - benchmark_results/**/*.json (recursive fallback for any JSON files)

    Returns dict mapping game_type -> list of episode dicts.
    """
    seen_ids: set[str] = set()  # Dedupe by episode_id
    episodes: dict[str, list[dict]] = {
        "codenames": [],
        "decrypto": [],
        "hanabi": [],
    }

    def add_episode(data: dict, filename: str) -> None:
        """Add episode if not already seen."""
        ep_id = data.get("episode_id", "")
        if ep_id and ep_id in seen_ids:
            return
        if ep_id:
            seen_ids.add(ep_id)

        game_type = _detect_game_type(data, filename)
        if game_type:
            episodes[game_type].append(data)

    bench_dir = _benchmark_results_dir()
    if not bench_dir.exists():
        return episodes

    # 1. Scan sessions/{game_type}/ (UI sessions)
    sessions_dir = _sessions_dir()
    for game_type in episodes.keys():
        game_dir = sessions_dir / game_type
        if game_dir.exists():
            for path in game_dir.glob("*.json"):
                data = _load_episode_file(path)
                if data:
                    add_episode(data, path.name)

    # 2. Scan experiment directories
    for exp_dir in bench_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ("sessions", "lost+found"):
            continue

        # Pattern A: */episodes/ (flat episodes directory)
        episodes_dir = exp_dir / "episodes"
        if episodes_dir.exists():
            for path in episodes_dir.glob("*.json"):
                data = _load_episode_file(path)
                if data:
                    add_episode(data, path.name)

        # Pattern B: */{game_type}/episodes/ (cloud benchmark structure)
        for game_type in ("codenames", "decrypto", "hanabi"):
            game_eps_dir = exp_dir / game_type / "episodes"
            if game_eps_dir.exists():
                for path in game_eps_dir.glob("*.json"):
                    data = _load_episode_file(path)
                    if data:
                        add_episode(data, path.name)

    return episodes


def compute_model_stats(episodes: dict[str, list[dict]]) -> dict[str, ModelStats]:
    """
    Compute aggregated stats for each model across all games.
    """
    stats: dict[str, ModelStats] = defaultdict(lambda: ModelStats(model=""))

    # Process Codenames
    for ep in episodes.get("codenames", []):
        for model, won in _extract_codenames_models(ep):
            if model not in stats:
                stats[model] = ModelStats(model=model)
            stats[model].codenames_games += 1
            if won:
                stats[model].codenames_wins += 1

    # Process Decrypto
    for ep in episodes.get("decrypto", []):
        for model, won in _extract_decrypto_models(ep):
            if model not in stats:
                stats[model] = ModelStats(model=model)
            stats[model].decrypto_games += 1
            if won:
                stats[model].decrypto_wins += 1

    # Process Hanabi
    for ep in episodes.get("hanabi", []):
        for model, score in _extract_hanabi_models(ep):
            if model not in stats:
                stats[model] = ModelStats(model=model)
            stats[model].hanabi_games += 1
            stats[model].hanabi_total_score += score

    return dict(stats)


def compute_codenames_rankings(stats: dict[str, ModelStats]) -> list[CodenamesRanking]:
    """Compute Codenames-specific rankings."""
    rankings = []

    for model, s in stats.items():
        if s.codenames_games > 0:
            win_rate = s.codenames_wins / s.codenames_games
            rankings.append(CodenamesRanking(
                model=_normalize_model_name(model),
                games=s.codenames_games,
                wins=s.codenames_wins,
                win_rate=round(win_rate, 3),
            ))

    # Sort by win rate descending
    rankings.sort(key=lambda x: (-x.win_rate, -x.games))
    return rankings


def compute_decrypto_rankings(stats: dict[str, ModelStats]) -> list[DecryptoRanking]:
    """Compute Decrypto-specific rankings."""
    rankings = []

    for model, s in stats.items():
        if s.decrypto_games > 0:
            win_rate = s.decrypto_wins / s.decrypto_games
            rankings.append(DecryptoRanking(
                model=_normalize_model_name(model),
                games=s.decrypto_games,
                wins=s.decrypto_wins,
                win_rate=round(win_rate, 3),
            ))

    # Sort by win rate descending
    rankings.sort(key=lambda x: (-x.win_rate, -x.games))
    return rankings


def compute_hanabi_rankings(stats: dict[str, ModelStats]) -> list[HanabiRanking]:
    """Compute Hanabi-specific rankings."""
    rankings = []

    for model, s in stats.items():
        if s.hanabi_games > 0:
            avg_score = s.hanabi_total_score / s.hanabi_games
            score_pct = (avg_score / 25) * 100
            rankings.append(HanabiRanking(
                model=_normalize_model_name(model),
                games=s.hanabi_games,
                avg_score=round(avg_score, 1),
                score_pct=round(score_pct, 1),
            ))

    # Sort by average score descending
    rankings.sort(key=lambda x: (-x.avg_score, -x.games))
    return rankings


def compute_overall_rankings(stats: dict[str, ModelStats]) -> list[OverallRanking]:
    """
    Compute overall rankings using normalized scores.

    Each game type contributes equally (33.3% each).
    Score is normalized: win_rate * 100 for competitive games, score_pct for Hanabi.
    """
    rankings = []

    for model, s in stats.items():
        total_games = s.codenames_games + s.decrypto_games + s.hanabi_games
        if total_games == 0:
            continue

        # Compute per-game scores (0-100 scale)
        codenames_score = None
        decrypto_score = None
        hanabi_score = None

        if s.codenames_games > 0:
            codenames_score = (s.codenames_wins / s.codenames_games) * 100

        if s.decrypto_games > 0:
            decrypto_score = (s.decrypto_wins / s.decrypto_games) * 100

        if s.hanabi_games > 0:
            avg_score = s.hanabi_total_score / s.hanabi_games
            hanabi_score = (avg_score / 25) * 100

        # Compute overall as average of available scores
        scores = [s for s in [codenames_score, decrypto_score, hanabi_score] if s is not None]
        overall = sum(scores) / len(scores) if scores else 0

        rankings.append(OverallRanking(
            rank=0,  # Will be set after sorting
            model=_normalize_model_name(model),
            games_played=total_games,
            overall_score=round(overall, 1),
            codenames_score=round(codenames_score, 1) if codenames_score else None,
            decrypto_score=round(decrypto_score, 1) if decrypto_score else None,
            hanabi_score=round(hanabi_score, 1) if hanabi_score else None,
        ))

    # Sort by overall score descending
    rankings.sort(key=lambda x: (-x.overall_score, -x.games_played))

    # Assign ranks
    for i, r in enumerate(rankings):
        r.rank = i + 1

    return rankings


def build_leaderboard() -> LeaderboardData:
    """
    Build complete leaderboard from all available episodes.

    Returns LeaderboardData with all rankings computed.
    """
    # Scan episodes
    episodes = scan_all_episodes()

    # Compute stats
    stats = compute_model_stats(episodes)

    # Build rankings
    return LeaderboardData(
        generated_at=datetime.utcnow().isoformat() + "Z",
        total_episodes={
            "codenames": len(episodes["codenames"]),
            "decrypto": len(episodes["decrypto"]),
            "hanabi": len(episodes["hanabi"]),
        },
        overall_rankings=compute_overall_rankings(stats),
        codenames_rankings=compute_codenames_rankings(stats),
        decrypto_rankings=compute_decrypto_rankings(stats),
        hanabi_rankings=compute_hanabi_rankings(stats),
    )


def save_leaderboard(data: LeaderboardData) -> Path:
    """Save leaderboard data to benchmark_results/leaderboard.json."""
    path = _benchmark_results_dir() / "leaderboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data.model_dump(mode="json"), f, indent=2)

    return path


def load_leaderboard() -> LeaderboardData | None:
    """Load leaderboard from disk if it exists."""
    path = _benchmark_results_dir() / "leaderboard.json"
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    return LeaderboardData.model_validate(data)


if __name__ == "__main__":
    # Quick test
    lb = build_leaderboard()
    print(f"Generated leaderboard with {lb.total_episodes} episodes")
    print(f"Overall rankings: {len(lb.overall_rankings)} models")
    for r in lb.overall_rankings[:5]:
        print(f"  #{r.rank} {r.model}: {r.overall_score}%")

    path = save_leaderboard(lb)
    print(f"Saved to {path}")
