from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TeamKey = Literal["red", "blue"]


class RoundCounters(BaseModel):
    """Public counters used for win/loss + state-slicing."""

    model_config = ConfigDict(frozen=True)

    own_interceptions: int
    own_miscommunications: int
    opp_interceptions: int
    opp_miscommunications: int


class RoundStateTag(BaseModel):
    """State tags evaluated at clue time (counters_before)."""

    model_config = ConfigDict(frozen=True)

    interceptions_state: Literal["leading", "trailing", "tied"]
    danger: bool


class CodeTriple(BaseModel):
    """A 3-digit Decrypto code: 3 distinct digits from {1,2,3,4}."""

    model_config = ConfigDict(frozen=True)

    digits: tuple[int, int, int]

    @staticmethod
    def validate_digits(d: tuple[int, int, int]) -> None:
        if len(d) != 3:
            raise ValueError("Code must have exactly 3 digits")
        if any(x not in (1, 2, 3, 4) for x in d):
            raise ValueError("Code digits must be in {1,2,3,4}")
        if len(set(d)) != 3:
            raise ValueError("Code digits must be distinct")

    @classmethod
    def from_list(cls, digits: list[int]) -> "CodeTriple":
        t = tuple(int(x) for x in digits)
        if len(t) != 3:
            raise ValueError("Code must have exactly 3 digits")
        cls.validate_digits(t)  # type: ignore[arg-type]
        return cls(digits=t)  # type: ignore[arg-type]


class CluerAnnotations(BaseModel):
    """
    Private cluer annotations. Must never be fed back into other agents.
    """

    model_config = ConfigDict(frozen=True)

    intended_mapping: dict[str, str] | None = None  # digit -> key word
    clue_rationale: dict[str, str] | None = None  # clue -> key word
    predicted_team_guess: list[int] | None = None
    p_team_correct: float | None = None
    p_intercept: float | None = None
    # New ToM predictions: distributions over the 24 possible codes (keys are "1-2-3")
    opponent_decode_dist: dict[str, float] | None = None
    opponent_intercept_dist: dict[str, float] | None = None


class ClueSet(BaseModel):
    """Public clue set: exactly 3 clues in order."""

    model_config = ConfigDict(frozen=True)

    clues: tuple[str, str, str]


class GuesserIndependent(BaseModel):
    """Independent guess prior to discussion."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    guess: tuple[int, int, int] | None  # None if parse/repair failed
    confidence: float | None  # required when guess is present; 0..1
    rationale: str
    parse_ok: bool
    grounding_ok: bool = True
    overconfident: bool = False
    slot_hypotheses: dict[str, str] | None = None  # digit(str) -> short theme label (optional)
    mapping_references: list["MappingReference"] = Field(default_factory=list)
    mapping_labels_ok: bool = True
    parse_error: str | None = None
    parse_retry_used: bool = False


class GuesserShare(BaseModel):
    """Bounded 1-message rationale share step."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    message: str


class ConsensusGuess(BaseModel):
    """Final consensus for decode or intercept."""

    model_config = ConfigDict(frozen=True)

    captain_id: str
    guess: tuple[int, int, int] | None
    confidence: float | None
    rationale: str
    parse_ok: bool
    grounding_ok: bool = True
    overconfident: bool = False
    slot_hypotheses: dict[str, str] | None = None
    mapping_references: list["MappingReference"] = Field(default_factory=list)
    mapping_labels_ok: bool = True
    parse_error: str | None = None
    parse_retry_used: bool = False


class ActionLog(BaseModel):
    """Self-contained log for one guess action (decode or intercept)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["decode", "intercept"]
    team: TeamKey  # the team producing this action (decoding own or intercepting opponent)
    opponent_team: TeamKey
    independent: tuple[GuesserIndependent, GuesserIndependent]
    share: tuple[GuesserShare, GuesserShare]
    consensus: ConsensusGuess
    correct: bool
    confirmed_mapping_count: int = 0
    uninformed: bool = False


class MappingReference(BaseModel):
    """
    A mapping claim referenced in reasoning.

    status semantics:
      - confirmed: must be grounded in a prior reveal (digit<->clue pairing)
      - hypothesis: agent belief, not yet confirmed
      - eliminated: agent belief that mapping is unlikely/ruled out
    """

    model_config = ConfigDict(frozen=True)

    mapping_type: Literal["digit_clue", "digit_theme"]
    digit: Literal["1", "2", "3", "4"]
    value: str  # clue word (digit_clue) or theme label (digit_theme)
    status: Literal["confirmed", "hypothesis", "eliminated"]
    support: str | None = None  # short justification / evidence pointer (optional)

    @model_validator(mode="before")
    @classmethod
    def _backward_compat(cls, data: Any) -> Any:
        """
        Backward compatibility:
        - Old schema used {digit, clue, status, support?} with implicit digit_clue.
        - Accept it and convert to {mapping_type='digit_clue', value=<clue>}.
        """
        if isinstance(data, dict):
            if "mapping_type" not in data:
                # default to digit_clue
                data = dict(data)
                data["mapping_type"] = "digit_clue"
            if "value" not in data and "clue" in data:
                data = dict(data)
                data["value"] = data.get("clue")
        return data


class RoundLog(BaseModel):
    """Self-contained per-round log (no hidden state leaks)."""

    model_config = ConfigDict(frozen=True)

    round_number: int
    counters_before: dict[TeamKey, RoundCounters]
    counters_after: dict[TeamKey, RoundCounters]
    round_state_at_clue_time: dict[TeamKey, RoundStateTag]

    # Public: both clue sets
    public_clues: dict[TeamKey, ClueSet]

    # Reveal: both true codes (revealed at end of round)
    reveal_true_codes: dict[TeamKey, tuple[int, int, int]]

    # All four actions (2 decode + 2 intercept)
    actions: tuple[ActionLog, ActionLog, ActionLog, ActionLog]

    # Structured state snapshots extracted from agent outputs (not inferred from free text).
    round_state: dict[TeamKey, dict[str, Any]] = Field(default_factory=dict)

    # Private: cluer annotations (never used to build future views)
    private: dict[TeamKey, dict[str, Any]] = Field(default_factory=dict)


class RoundInputs(BaseModel):
    """
    Frozen snapshot used to build all agent views for a round.

    Critical: this object must not include cluer annotations, RNG objects, or debug state.
    """

    model_config = ConfigDict(frozen=True)

    game_id: str
    seed: int
    round_number: int

    # Keys (private by team; visibility filters decide what to show)
    keys: dict[TeamKey, tuple[str, str, str, str]]

    # Current codes (private by team; must not be visible to guessers)
    current_codes: dict[TeamKey, tuple[int, int, int]]

    # Public history up to end of previous round (already revealed)
    history_rounds: tuple[RoundLog, ...]

    # Public counters at start of round (used for state tags + displayed to all)
    counters_before: dict[TeamKey, RoundCounters]

    # Public clues for this round (filled after cluer phase)
    public_clues: dict[TeamKey, ClueSet] | None = None


class DecryptoConfig(BaseModel):
    """Config for a Decrypto game."""

    max_rounds: int = 8
    keyword_bank_path: str | None = None  # default to data/wordlist.txt
    seed: int = 0


class DecryptoEpisodeRecord(BaseModel):
    """Complete episode record for Decrypto (standalone)."""

    episode_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: DecryptoConfig
    game_id: str
    seed: int
    keys: dict[TeamKey, tuple[str, str, str, str]]
    code_sequences: dict[TeamKey, tuple[tuple[int, int, int], ...]]
    rounds: tuple[RoundLog, ...]
    winner: TeamKey | None = None
    result_reason: Literal["interceptions", "miscommunications", "survived", "max_rounds", "tie_interceptions", "tie_miscommunications"] | None = None
    scores: dict[str, Any] = Field(default_factory=dict)

    def to_filename(self) -> str:
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"decrypto_episode_{self.episode_id}_{ts}.json"

    def save(self, directory: str) -> str:
        """
        Save episode JSON to a directory. Returns the written filepath.
        """
        from pathlib import Path
        import json

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        fp = d / self.to_filename()
        data = self.model_dump(mode="json")
        data["timestamp"] = self.timestamp.isoformat()
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)
        return str(fp)

