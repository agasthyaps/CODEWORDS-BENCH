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
    decrypto_decode_attempts: int = 0
    decrypto_decode_successes: int = 0
    decrypto_intercept_attempts: int = 0
    decrypto_intercept_successes: int = 0
    # Hanabi
    hanabi_games: int = 0
    hanabi_total_score: int = 0
    hanabi_total_turns: int = 0
    hanabi_turn_limit_hits: int = 0


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
    decode_accuracy: float = 0.0  # Teammate understanding
    intercept_accuracy: float = 0.0  # Opponent modeling (pure ToM)


class HanabiRanking(BaseModel):
    """Hanabi ranking entry."""

    model: str
    games: int
    avg_score: float
    score_pct: float  # As percentage of max 25
    # Efficiency metrics (the key insight from research)
    efficiency: float = 0.0  # score/turn - measures true cooperative ToM
    avg_turns: float = 0.0
    turn_limit_pct: float = 0.0  # % games hitting turn limit


class OverallRanking(BaseModel):
    """Overall model ranking."""

    rank: int
    model: str
    games_played: int
    overall_score: float  # 0-100 composite (efficiency-based)
    # Per-game scores (0-100)
    codenames_score: float | None = None
    decrypto_score: float | None = None
    hanabi_score: float | None = None  # Efficiency-based
    # Alternative: raw score composite (for comparison toggle)
    raw_overall_score: float = 0.0
    raw_hanabi_score: float | None = None  # Raw avg score as pct
    # Detailed metrics for tooltip/display
    hanabi_efficiency: float | None = None
    decrypto_decode: float | None = None
    decrypto_intercept: float | None = None


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


def _extract_decrypto_models(episode: dict) -> list[tuple[str, bool, int, int, int, int]]:
    """
    Extract models from a Decrypto episode.

    Returns list of (model, won, decode_attempts, decode_successes, intercept_attempts, intercept_successes) tuples.
    """
    metadata = episode.get("metadata", {})
    winner = (episode.get("winner") or "").lower()
    results = []

    # Parse round results to get decode/intercept stats
    rounds = episode.get("rounds", [])

    # Track per-team stats
    red_decode_attempts = 0
    red_decode_successes = 0
    red_intercept_attempts = 0
    red_intercept_successes = 0
    blue_decode_attempts = 0
    blue_decode_successes = 0
    blue_intercept_attempts = 0
    blue_intercept_successes = 0

    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue

        # Each round has decode and intercept results for both teams
        # Format varies - try multiple approaches
        actions = round_data.get("actions", {})

        # Handle case where actions is a list (iterate to find relevant items)
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, dict):
                    continue
                # Check multiple possible field names for action type
                action_type = (action.get("kind") or action.get("type") or action.get("action_type") or "").lower()
                team = (action.get("team") or "").lower()
                correct = action.get("correct") or action.get("success")

                if action_type == "decode":
                    if team == "red":
                        red_decode_attempts += 1
                        if correct:
                            red_decode_successes += 1
                    elif team == "blue":
                        blue_decode_attempts += 1
                        if correct:
                            blue_decode_successes += 1
                elif action_type == "intercept":
                    if team == "red":
                        red_intercept_attempts += 1
                        if correct:
                            red_intercept_successes += 1
                    elif team == "blue":
                        blue_intercept_attempts += 1
                        if correct:
                            blue_intercept_successes += 1
        elif isinstance(actions, dict):
            # Check for decode results
            red_decode = actions.get("red_decode") or round_data.get("red_decode_correct")
            blue_decode = actions.get("blue_decode") or round_data.get("blue_decode_correct")

            if red_decode is not None:
                red_decode_attempts += 1
                if red_decode is True or red_decode == "correct":
                    red_decode_successes += 1

            if blue_decode is not None:
                blue_decode_attempts += 1
                if blue_decode is True or blue_decode == "correct":
                    blue_decode_successes += 1

            # Check for intercept results
            red_intercept = actions.get("red_intercept") or round_data.get("red_intercept_correct")
            blue_intercept = actions.get("blue_intercept") or round_data.get("blue_intercept_correct")

            if red_intercept is not None:
                red_intercept_attempts += 1
                if red_intercept is True or red_intercept == "correct":
                    red_intercept_successes += 1

            if blue_intercept is not None:
                blue_intercept_attempts += 1
                if blue_intercept is True or blue_intercept == "correct":
                    blue_intercept_successes += 1

        # Also check round-level decode/intercept results (alternate format)
        for team, prefix in [("red", "red_"), ("blue", "blue_")]:
            decode_key = f"{prefix}decode_correct"
            intercept_key = f"{prefix}intercept_correct"

            if decode_key in round_data and round_data[decode_key] is not None:
                if team == "red" and red_decode_attempts == 0:
                    red_decode_attempts += 1
                    if round_data[decode_key]:
                        red_decode_successes += 1
                elif team == "blue" and blue_decode_attempts == 0:
                    blue_decode_attempts += 1
                    if round_data[decode_key]:
                        blue_decode_successes += 1

            if intercept_key in round_data and round_data[intercept_key] is not None:
                if team == "red" and red_intercept_attempts == 0:
                    red_intercept_attempts += 1
                    if round_data[intercept_key]:
                        red_intercept_successes += 1
                elif team == "blue" and blue_intercept_attempts == 0:
                    blue_intercept_attempts += 1
                    if round_data[intercept_key]:
                        blue_intercept_successes += 1

    # If no decode/intercept data parsed, we can't compute accuracy
    # Just leave them at 0 (will show as 0% accuracy, which is honest)

    # Red team
    red_team = metadata.get("red_team", {})
    red_cluer = red_team.get("cluer_model")
    if red_cluer:
        red_won = winner == "red"
        results.append((red_cluer, red_won, red_decode_attempts, red_decode_successes,
                       red_intercept_attempts, red_intercept_successes))

    # Blue team
    blue_team = metadata.get("blue_team", {})
    blue_cluer = blue_team.get("cluer_model")
    if blue_cluer:
        blue_won = winner == "blue"
        results.append((blue_cluer, blue_won, blue_decode_attempts, blue_decode_successes,
                       blue_intercept_attempts, blue_intercept_successes))

    return results


def _extract_hanabi_models(episode: dict) -> list[tuple[str, int, int, bool]]:
    """
    Extract models from a Hanabi episode.

    Returns list of (model, score, turns, hit_turn_limit) tuples.
    Since Hanabi is cooperative with same model for all players,
    we return just one entry per unique model.
    """
    metadata = episode.get("metadata", {}) or {}
    score = episode.get("final_score") or 0

    # Get turn count - check multiple locations
    turns = len(episode.get("turns", []))
    if turns == 0:
        turns = metadata.get("total_turns", 0) or episode.get("total_turns", 0)

    # Determine if game hit turn limit (vs ended by fuse-out or completion)
    game_over_reason = (episode.get("game_over_reason") or "").lower()
    hit_turn_limit = "turn" in game_over_reason or "limit" in game_over_reason
    # Also check if turns is very high (200+) as indicator
    if turns >= 200:
        hit_turn_limit = True

    # Try multiple locations for model info
    # 1. metadata.model (cloud benchmark format)
    model = metadata.get("model")
    if model:
        return [(model, score, turns, hit_turn_limit)]

    # 2. metadata.player_models (alternative format)
    player_models = metadata.get("player_models", {}) or {}
    if player_models:
        unique_models = set(player_models.values())
        return [(m, score, turns, hit_turn_limit) for m in unique_models if m]

    # 3. config.model (fallback)
    config = episode.get("config", {}) or {}
    model = config.get("model")
    if model:
        return [(model, score, turns, hit_turn_limit)]

    return []


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
    - benchmark_results/sessions/{game_type}/*.json (UI sessions)
    - benchmark_results/{game_type}/episodes/*.json (direct game type dirs)
    - benchmark_results/{run_name}/episodes/*.json (flat run structure)
    - benchmark_results/{run_name}/{game_type}/episodes/*.json (production structure)

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

    # 2. Scan {game_type}/episodes/ directly under benchmark_results (production structure)
    for game_type in ("codenames", "decrypto", "hanabi"):
        game_eps_dir = bench_dir / game_type / "episodes"
        if game_eps_dir.exists():
            for path in game_eps_dir.glob("*.json"):
                data = _load_episode_file(path)
                if data:
                    add_episode(data, path.name)

    # 3. Scan run directories (benchmark_results/{run_name}/{game_type}/episodes/)
    for exp_dir in bench_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ("sessions", "lost+found", "codenames", "decrypto", "hanabi"):
            continue

        # Pattern: */episodes/ (flat episodes directory)
        episodes_dir = exp_dir / "episodes"
        if episodes_dir.exists():
            for path in episodes_dir.glob("*.json"):
                data = _load_episode_file(path)
                if data:
                    add_episode(data, path.name)

        # Also check */{game_type}/episodes/ (nested cloud benchmark structure)
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
        for model, won, decode_att, decode_succ, intercept_att, intercept_succ in _extract_decrypto_models(ep):
            if model not in stats:
                stats[model] = ModelStats(model=model)
            stats[model].decrypto_games += 1
            if won:
                stats[model].decrypto_wins += 1
            stats[model].decrypto_decode_attempts += decode_att
            stats[model].decrypto_decode_successes += decode_succ
            stats[model].decrypto_intercept_attempts += intercept_att
            stats[model].decrypto_intercept_successes += intercept_succ

    # Process Hanabi
    for ep in episodes.get("hanabi", []):
        for model, score, turns, hit_limit in _extract_hanabi_models(ep):
            if model not in stats:
                stats[model] = ModelStats(model=model)
            stats[model].hanabi_games += 1
            stats[model].hanabi_total_score += int(score) if score else 0
            stats[model].hanabi_total_turns += int(turns) if turns else 0
            if hit_limit:
                stats[model].hanabi_turn_limit_hits += 1

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

            # Compute decode and intercept accuracy
            decode_acc = 0.0
            if s.decrypto_decode_attempts > 0:
                decode_acc = s.decrypto_decode_successes / s.decrypto_decode_attempts

            intercept_acc = 0.0
            if s.decrypto_intercept_attempts > 0:
                intercept_acc = s.decrypto_intercept_successes / s.decrypto_intercept_attempts

            rankings.append(DecryptoRanking(
                model=_normalize_model_name(model),
                games=s.decrypto_games,
                wins=s.decrypto_wins,
                win_rate=round(win_rate, 3),
                decode_accuracy=round(decode_acc, 3),
                intercept_accuracy=round(intercept_acc, 3),
            ))

    # Sort by win rate descending (could also sort by intercept for "pure ToM")
    rankings.sort(key=lambda x: (-x.win_rate, -x.games))
    return rankings


def compute_hanabi_rankings(stats: dict[str, ModelStats]) -> list[HanabiRanking]:
    """Compute Hanabi-specific rankings.

    Key insight from research: efficiency (score/turn) is more meaningful than raw score.
    GPT-5.2 has highest raw score but lowest efficiency due to hitting turn limit.
    """
    rankings = []

    for model, s in stats.items():
        if s.hanabi_games > 0:
            avg_score = s.hanabi_total_score / s.hanabi_games
            score_pct = (avg_score / 25) * 100

            # Efficiency metrics
            avg_turns = s.hanabi_total_turns / s.hanabi_games if s.hanabi_total_turns > 0 else 0
            efficiency = avg_score / avg_turns if avg_turns > 0 else 0
            turn_limit_pct = (s.hanabi_turn_limit_hits / s.hanabi_games) * 100

            rankings.append(HanabiRanking(
                model=_normalize_model_name(model),
                games=s.hanabi_games,
                avg_score=round(avg_score, 1),
                score_pct=round(score_pct, 1),
                efficiency=round(efficiency, 3),
                avg_turns=round(avg_turns, 1),
                turn_limit_pct=round(turn_limit_pct, 1),
            ))

    # Sort by EFFICIENCY descending (not raw score!) - this is the key research insight
    rankings.sort(key=lambda x: (-x.efficiency, -x.games))
    return rankings


def compute_overall_rankings(stats: dict[str, ModelStats]) -> list[OverallRanking]:
    """
    Compute overall rankings using normalized scores.

    Key research insight: We compute TWO composites:
    1. Efficiency-based (default): Uses Hanabi efficiency, not raw score
    2. Raw score-based: Uses traditional raw Hanabi score (for comparison)

    This allows the UI to toggle between views and show how rankings flip.
    """
    rankings = []

    # First pass: compute all metrics to find max values for normalization
    hanabi_efficiencies = []
    for model, s in stats.items():
        if s.hanabi_games > 0 and s.hanabi_total_turns > 0:
            avg_turns = s.hanabi_total_turns / s.hanabi_games
            avg_score = s.hanabi_total_score / s.hanabi_games
            eff = avg_score / avg_turns if avg_turns > 0 else 0
            hanabi_efficiencies.append(eff)

    max_efficiency = max(hanabi_efficiencies) if hanabi_efficiencies else 1.0

    for model, s in stats.items():
        total_games = s.codenames_games + s.decrypto_games + s.hanabi_games
        if total_games == 0:
            continue

        # Compute per-game scores (0-100 scale)
        codenames_score = None
        decrypto_score = None

        if s.codenames_games > 0:
            codenames_score = (s.codenames_wins / s.codenames_games) * 100

        if s.decrypto_games > 0:
            decrypto_score = (s.decrypto_wins / s.decrypto_games) * 100

        # Hanabi: compute both efficiency-based and raw score
        hanabi_efficiency_score = None
        raw_hanabi_score = None
        hanabi_efficiency_val = None
        if s.hanabi_games > 0:
            avg_score = s.hanabi_total_score / s.hanabi_games
            raw_hanabi_score = (avg_score / 25) * 100

            if s.hanabi_total_turns > 0:
                avg_turns = s.hanabi_total_turns / s.hanabi_games
                efficiency = avg_score / avg_turns if avg_turns > 0 else 0
                hanabi_efficiency_val = efficiency
                # Normalize efficiency to 0-100 scale
                hanabi_efficiency_score = (efficiency / max_efficiency) * 100 if max_efficiency > 0 else 0

        # Decrypto detailed metrics
        decode_acc = None
        intercept_acc = None
        if s.decrypto_decode_attempts > 0:
            decode_acc = (s.decrypto_decode_successes / s.decrypto_decode_attempts) * 100
        if s.decrypto_intercept_attempts > 0:
            intercept_acc = (s.decrypto_intercept_successes / s.decrypto_intercept_attempts) * 100

        # Compute EFFICIENCY-BASED overall (using hanabi efficiency, not raw score)
        eff_scores = [s for s in [codenames_score, decrypto_score, hanabi_efficiency_score] if s is not None]
        overall_eff = sum(eff_scores) / len(eff_scores) if eff_scores else 0

        # Compute RAW-BASED overall (using hanabi raw score)
        raw_scores = [s for s in [codenames_score, decrypto_score, raw_hanabi_score] if s is not None]
        overall_raw = sum(raw_scores) / len(raw_scores) if raw_scores else 0

        rankings.append(OverallRanking(
            rank=0,  # Will be set after sorting
            model=_normalize_model_name(model),
            games_played=total_games,
            # Efficiency-based (default)
            overall_score=round(overall_eff, 1),
            codenames_score=round(codenames_score, 1) if codenames_score else None,
            decrypto_score=round(decrypto_score, 1) if decrypto_score else None,
            hanabi_score=round(hanabi_efficiency_score, 1) if hanabi_efficiency_score else None,
            # Raw-based (for comparison toggle)
            raw_overall_score=round(overall_raw, 1),
            raw_hanabi_score=round(raw_hanabi_score, 1) if raw_hanabi_score else None,
            # Detailed metrics
            hanabi_efficiency=round(hanabi_efficiency_val, 3) if hanabi_efficiency_val else None,
            decrypto_decode=round(decode_acc, 1) if decode_acc else None,
            decrypto_intercept=round(intercept_acc, 1) if intercept_acc else None,
        ))

    # Sort by EFFICIENCY-BASED overall score descending (research-aligned default)
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
