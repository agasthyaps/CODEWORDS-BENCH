"""LLM provider abstraction for agent interactions.

This module re-exports from src.core.llm for backward compatibility.
New code should import directly from src.core.
"""

from src.core.llm import (
    LLMProvider,
    LLMResponse,
    OpenRouterProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
    create_provider,
)

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenRouterProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "create_provider",
]
