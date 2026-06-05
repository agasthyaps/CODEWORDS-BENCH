"""Core module with shared abstractions for all games."""

from .state import AgentState, AgentStateManager
from .trace import AgentTrace
from .parsing import extract_scratchpad, remove_scratchpad_from_response
from .benchmarking import (
    BenchmarkCondition,
    BenchmarkConditionName,
    HarnessEvent,
    BenchmarkPenalty,
    RunManifest,
    default_benchmark_condition,
    resolve_condition,
    build_run_manifest,
)
from .llm import (
    LLMProvider,
    LLMResponse,
    OpenRouterProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
    create_provider,
)

__all__ = [
    # State management
    "AgentState",
    "AgentStateManager",
    # Tracing
    "AgentTrace",
    # Parsing
    "extract_scratchpad",
    "remove_scratchpad_from_response",
    # Benchmark protocol
    "BenchmarkCondition",
    "BenchmarkConditionName",
    "HarnessEvent",
    "BenchmarkPenalty",
    "RunManifest",
    "default_benchmark_condition",
    "resolve_condition",
    "build_run_manifest",
    # LLM providers
    "LLMProvider",
    "LLMResponse",
    "OpenRouterProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "create_provider",
]
