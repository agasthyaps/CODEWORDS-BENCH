"""Shared benchmark protocol models for humanlike-condition runs."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class BenchmarkConditionName(str, Enum):
    HUMAN_TABLE_SCRATCHPAD = "human_table_scratchpad"
    RAW_CHAT = "raw_chat"
    STRUCTURED_OUTPUT = "structured_output"


ScratchpadPolicy = Literal["private", "none"]
RepairPolicy = Literal["neutral_format_and_legality", "none", "structured_output"]
InvalidActionPolicy = Literal["human_table", "record_only", "schema_enforced"]
StructuredOutputPolicy = Literal["disabled", "provider_schema"]


class BenchmarkCondition(BaseModel):
    """Named experimental condition for a benchmark run."""

    name: BenchmarkConditionName = BenchmarkConditionName.HUMAN_TABLE_SCRATCHPAD
    scratchpad_policy: ScratchpadPolicy = "private"
    repair_policy: RepairPolicy = "neutral_format_and_legality"
    invalid_action_policy: InvalidActionPolicy = "human_table"
    structured_output_policy: StructuredOutputPolicy = "disabled"
    description: str = (
        "Human-table default: private notes, neutral repairs, and table moderation."
    )

    @property
    def scratchpad_enabled(self) -> bool:
        return self.scratchpad_policy == "private"

    @property
    def repair_enabled(self) -> bool:
        return self.repair_policy != "none"


CONDITIONS: dict[BenchmarkConditionName, BenchmarkCondition] = {
    BenchmarkConditionName.HUMAN_TABLE_SCRATCHPAD: BenchmarkCondition(),
    BenchmarkConditionName.RAW_CHAT: BenchmarkCondition(
        name=BenchmarkConditionName.RAW_CHAT,
        scratchpad_policy="none",
        repair_policy="none",
        invalid_action_policy="record_only",
        structured_output_policy="disabled",
        description="Diagnostic raw chat: no private memory and no format repair.",
    ),
    BenchmarkConditionName.STRUCTURED_OUTPUT: BenchmarkCondition(
        name=BenchmarkConditionName.STRUCTURED_OUTPUT,
        scratchpad_policy="private",
        repair_policy="structured_output",
        invalid_action_policy="schema_enforced",
        structured_output_policy="provider_schema",
        description="Comparator condition for provider/schema-enforced outputs.",
    ),
}


def default_benchmark_condition() -> BenchmarkCondition:
    return CONDITIONS[BenchmarkConditionName.HUMAN_TABLE_SCRATCHPAD].model_copy(deep=True)


def resolve_condition(
    condition: BenchmarkCondition | BenchmarkConditionName | str | None,
) -> BenchmarkCondition:
    """Resolve a condition name or model to a full BenchmarkCondition."""
    if condition is None:
        return default_benchmark_condition()
    if isinstance(condition, BenchmarkCondition):
        return condition
    if isinstance(condition, BenchmarkConditionName):
        return CONDITIONS[condition].model_copy(deep=True)
    name = BenchmarkConditionName(str(condition))
    return CONDITIONS[name].model_copy(deep=True)


class HarnessEvent(BaseModel):
    """Typed event emitted by the harness around model and moderator actions."""

    event_type: Literal[
        "model_prompt_sent",
        "raw_model_response_received",
        "parse_succeeded",
        "parse_failed",
        "repair_prompt_issued",
        "fallback_action_used",
        "invalid_action_proposed",
        "human_table_correction_applied",
        "final_accepted_game_action",
        "turn_completed",
        "round_completed",
        "game_completed",
    ]
    game_type: str
    episode_id: str | None = None
    turn_number: int | None = None
    round_number: int | None = None
    team: str | None = None
    agent_id: str | None = None
    role: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkPenalty(BaseModel):
    """Penalty record for benchmark performance separate from game score."""

    penalty_type: Literal[
        "parse_failure",
        "repair_prompt",
        "fallback_action",
        "invalid_action",
        "moderator_correction",
        "transport_retry",
    ]
    game_type: str
    points: float = 1.0
    description: str
    episode_id: str | None = None
    turn_number: int | None = None
    round_number: int | None = None
    team: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Machine-readable reproducibility manifest saved with each run."""

    condition: BenchmarkCondition = Field(default_factory=default_benchmark_condition)
    game_type: str
    game_rules_version: str = "v1"
    code_version: str | None = None
    dirty_worktree: bool | None = None
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    provider_parameters: dict[str, Any] = Field(default_factory=dict)
    seed_schedule: list[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def _collect_prompt_hashes_cached(repo_root_str: str) -> tuple[tuple[str, str], ...]:
    repo_root = Path(repo_root_str)
    prompt_hashes: dict[str, str] = {}
    for prompt in sorted((repo_root / "src").glob("**/prompts/*.md")):
        rel = prompt.relative_to(repo_root).as_posix()
        prompt_hashes[rel] = hashlib.sha256(prompt.read_bytes()).hexdigest()
    return tuple(prompt_hashes.items())


def collect_prompt_hashes(root: Path | None = None) -> dict[str, str]:
    """Hash prompt files so prompt drift is visible in saved artifacts."""
    repo_root = root or Path(__file__).resolve().parents[2]
    return dict(_collect_prompt_hashes_cached(str(repo_root)))


def git_metadata(root: Path | None = None) -> tuple[str | None, bool | None]:
    """Return (git SHA, dirty flag), falling back cleanly outside git."""
    repo_root = root or Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return sha, bool(status.strip())
    except Exception:
        return None, None


def build_run_manifest(
    *,
    game_type: str,
    condition: BenchmarkCondition | BenchmarkConditionName | str | None = None,
    models: dict[str, Any] | None = None,
    provider_parameters: dict[str, Any] | None = None,
    seed_schedule: list[int] | None = None,
    game_rules_version: str = "v1",
) -> RunManifest:
    """Build a reproducibility manifest for a game episode or benchmark run."""
    sha, dirty = git_metadata()
    return RunManifest(
        condition=resolve_condition(condition),
        game_type=game_type,
        game_rules_version=game_rules_version,
        code_version=sha,
        dirty_worktree=dirty,
        prompt_hashes=collect_prompt_hashes(),
        model_metadata=models or {},
        provider_parameters=provider_parameters or {},
        seed_schedule=seed_schedule or [],
    )


PILOT_HOMOGENEOUS_SEEDS = list(range(5))
PILOT_MIXED_SEEDS = list(range(3))
MAIN_HOMOGENEOUS_SEEDS = list(range(30))
MAIN_MIXED_SEEDS = list(range(10))
