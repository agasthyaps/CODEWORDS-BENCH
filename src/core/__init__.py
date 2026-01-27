"""Core module with shared abstractions for all games."""

from .state import AgentState, AgentStateManager
from .trace import AgentTrace
from .parsing import extract_scratchpad, remove_scratchpad_from_response
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
    # LLM providers
    "LLMProvider",
    "LLMResponse",
    "OpenRouterProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "create_provider",
]
