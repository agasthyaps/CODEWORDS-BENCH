"""Storage helpers for UI sessions and replays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.runner.episode import ExtendedEpisodeRecord
from src.decrypto.models import DecryptoEpisodeRecord
from src.hanabi.models import HanabiEpisodeRecord

GameType = Literal["codenames", "decrypto", "hanabi"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_data_dir() -> Path:
    """Get benchmark data directory, using env var in production."""
    import os
    env_dir = os.environ.get("BENCHMARK_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return _repo_root() / "benchmark_results"


def _base_dir() -> Path:
    """Base directory for UI sessions - stored in benchmark_results for persistence."""
    return _benchmark_data_dir() / "sessions"


def _game_dir(game_type: GameType) -> Path:
    return _base_dir() / game_type


def ensure_storage() -> None:
    base = _base_dir()
    (base / "codenames").mkdir(parents=True, exist_ok=True)
    (base / "decrypto").mkdir(parents=True, exist_ok=True)
    (base / "hanabi").mkdir(parents=True, exist_ok=True)
    (base / "batches").mkdir(parents=True, exist_ok=True)
    (base / "stats").mkdir(parents=True, exist_ok=True)


def save_codenames_episode(episode: ExtendedEpisodeRecord) -> Path:
    ensure_storage()
    return episode.save(_game_dir("codenames"))


def save_decrypto_episode(episode: DecryptoEpisodeRecord) -> str:
    ensure_storage()
    return episode.save(str(_game_dir("decrypto")))


def save_hanabi_episode(episode: HanabiEpisodeRecord) -> str:
    ensure_storage()
    return episode.save(str(_game_dir("hanabi")))


def list_replays() -> list[dict[str, Any]]:
    ensure_storage()
    replays: list[dict[str, Any]] = []
    for game_type in ("codenames", "decrypto", "hanabi"):
        for path in sorted(_game_dir(game_type).glob("*.json")):
            replays.append(
                {
                    "replay_id": path.name,
                    "game_type": game_type,
                    "filename": path.name,
                }
            )
    return replays


def load_replay(game_type: GameType, replay_id: str) -> dict[str, Any]:
    path = _game_dir(game_type) / replay_id
    if not path.exists():
        raise FileNotFoundError(replay_id)
    with open(path, "r") as f:
        return json.load(f)


def save_stats_report(replay_id: str, report: dict[str, Any]) -> Path:
    ensure_storage()
    # Remove .json extension if present to avoid double extension
    base_name = replay_id.removesuffix(".json")
    stats_path = _base_dir() / "stats" / f"{base_name}.json"
    with open(stats_path, "w") as f:
        json.dump(report, f, indent=2)
    return stats_path


def load_stats_report(replay_id: str) -> dict[str, Any] | None:
    # Remove .json extension if present to avoid double extension
    base_name = replay_id.removesuffix(".json")
    stats_path = _base_dir() / "stats" / f"{base_name}.json"
    if not stats_path.exists():
        # Try with original name in case of old files
        stats_path = _base_dir() / "stats" / f"{replay_id}.json"
        if not stats_path.exists():
            return None
    with open(stats_path, "r") as f:
        return json.load(f)


def save_batch_log(batch_id: str, payload: dict[str, Any]) -> Path:
    ensure_storage()
    path = _base_dir() / "batches" / f"{batch_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_batch_log(batch_id: str) -> dict[str, Any] | None:
    path = _base_dir() / "batches" / f"{batch_id}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)
