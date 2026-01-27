"""LLM provider abstraction for agent interactions."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_response: dict[str, Any] | None = None
    reasoning_trace: str | None = None  # For reasoning models


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        pass


class OpenRouterProvider(LLMProvider):
    """LLM provider using OpenRouter API."""

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Generate a completion using OpenRouter with retry logic."""
        import asyncio
        import logging
        
        logger = logging.getLogger("llm")
        start_time = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=120.0,
                    )

                    if response.status_code != 200:
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("error", {}).get("message", response.text)
                        except Exception:
                            error_msg = response.text
                        
                        if response.status_code >= 500 or response.status_code == 429:
                            last_error = RuntimeError(f"OpenRouter API error ({response.status_code}): {error_msg}")
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt
                                logger.warning(f"API error {response.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                        raise RuntimeError(f"OpenRouter API error ({response.status_code}): {error_msg}")

                    data = response.json()
                    break
                    
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Network error ({type(e).__name__}), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"Network error after {max_retries} attempts: {e}") from e
        else:
            raise RuntimeError(f"Failed after {max_retries} attempts") from last_error

        latency_ms = (time.perf_counter() - start_time) * 1000

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw_response=data,
        )


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion using OpenAI API."""
        start_time = time.perf_counter()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw_response=data,
        )


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic API directly."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        start_time = time.perf_counter()

        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            request_body["system"] = system_message

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start_time) * 1000

        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            raw_response=data,
        )


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "mock-model",
    ):
        self.responses = responses or ["Mock response"]
        self.model = model
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a mock response."""
        self.last_messages = messages

        response_idx = self.call_count % len(self.responses)
        content = self.responses[response_idx]
        self.call_count += 1

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(content) // 4,
            latency_ms=10.0,
            raw_response=None,
        )


def create_provider(
    provider_type: str = "openrouter",
    model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> LLMProvider:
    """Factory function to create LLM providers."""
    providers = {
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Options: {list(providers.keys())}")

    provider_cls = providers[provider_type]

    provider_kwargs: dict[str, Any] = {}
    if model:
        provider_kwargs["model"] = model
    if api_key:
        provider_kwargs["api_key"] = api_key
    provider_kwargs.update(kwargs)

    return provider_cls(**provider_kwargs)
