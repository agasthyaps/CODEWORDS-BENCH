"""Agent trace models for logging interactions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    """Trace of a single agent interaction.
    
    Captures the full context of an LLM call including prompt,
    response, parsing results, and optional scratchpad/reasoning traces.
    """
    agent_id: str
    turn_number: int
    prompt_sent: str
    raw_response: str
    parsed_action: dict[str, Any] | None = None
    scratchpad_addition: str | None = None  # What agent added this turn
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    model: str
    temperature: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_trace: str | None = None  # For reasoning models (o1, extended thinking)
    
    # Prediction tracking for ToM metrics (Cluer only)
    predicted_success: float | None = None  # 0.0-1.0 confidence in guesser success
    predicted_targets: list[str] | None = None  # Expected guesser picks in order