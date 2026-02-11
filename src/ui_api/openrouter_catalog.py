"""Shared OpenRouter catalog helpers for UI and cost estimation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from src.benchmark.config import ModelConfig


_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _models_config_path() -> Path:
    return _repo_root() / "config" / "models.json"


def get_openrouter_base_url() -> str:
    """Get OpenRouter base URL from config/models.json with a sane default."""
    config_path = _models_config_path()
    if not config_path.exists():
        return _DEFAULT_OPENROUTER_BASE_URL

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        return str(data.get("openrouter_base_url") or _DEFAULT_OPENROUTER_BASE_URL)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return _DEFAULT_OPENROUTER_BASE_URL


def short_name_from_model_id(model_id: str) -> str:
    """Derive a compact display name from a model id."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def model_display_name(model: dict[str, Any]) -> str:
    """Pick the best available display name from OpenRouter payload."""
    for key in ("name", "canonical_name", "slug"):
        value = model.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_id = model.get("id")
    if isinstance(model_id, str) and model_id:
        return short_name_from_model_id(model_id)
    return "unknown"


def is_text_output_model(model: dict[str, Any]) -> bool:
    """Return True if the model appears to support text output.

    When modality data is missing, allow the model (safe fallback).
    """
    architecture = model.get("architecture")
    arch = architecture if isinstance(architecture, dict) else {}

    modalities: list[str] = []
    for value in (arch.get("modality"), model.get("modality")):
        if isinstance(value, str) and value.strip():
            modalities.append(value.strip().lower())

    if modalities:
        return any("->text" in modality or modality.endswith("text") for modality in modalities)

    output_modalities = arch.get("output_modalities") or model.get("output_modalities")
    if isinstance(output_modalities, list):
        normalized = {str(item).strip().lower() for item in output_modalities}
        return "text" in normalized

    # Safe fallback: treat missing modality metadata as text-capable.
    return True


def make_openrouter_model_config(model_id: str, *, name: str | None = None) -> ModelConfig:
    """Create a synthetic OpenRouter model config for model IDs not in models.json."""
    return ModelConfig(
        name=name or short_name_from_model_id(model_id),
        model_id=model_id,
        provider="openrouter",
        base_url=get_openrouter_base_url(),
    )


def resolve_model_config(model_map: dict[str, ModelConfig], model_id: str) -> ModelConfig:
    """Resolve a model from the curated map or synthesize an OpenRouter config."""
    model = model_map.get(model_id)
    if model is not None:
        return model
    return make_openrouter_model_config(model_id)


class OpenRouterCatalogCache:
    """Shared cached OpenRouter model catalog."""

    _instance: "OpenRouterCatalogCache | None" = None
    _models: list[dict[str, Any]] = []
    _last_fetch: float = 0
    _cache_ttl: float = 900  # 15 minutes

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_models(self, *, text_output_only: bool = False) -> list[dict[str, Any]]:
        """Get cached models and refresh from OpenRouter if stale."""
        await self._refresh_if_needed()

        models = self._models
        if text_output_only:
            models = [model for model in models if is_text_output_model(model)]

        return [dict(model) for model in models]

    async def refresh(self) -> None:
        """Force a catalog refresh from OpenRouter."""
        await self._fetch_models()

    async def _refresh_if_needed(self) -> None:
        if time.time() - self._last_fetch > self._cache_ttl:
            await self._fetch_models()

    async def _fetch_models(self) -> None:
        base_url = get_openrouter_base_url().rstrip("/")
        headers: dict[str, str] = {}
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{base_url}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()

        data = payload.get("data", [])
        if not isinstance(data, list):
            raise ValueError("OpenRouter /models response missing list 'data'")

        models: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(item)

        self._models = models
        self._last_fetch = time.time()


async def get_openrouter_model_infos(*, text_output_only: bool = True) -> list[dict[str, Any]]:
    """Return model objects shaped like API /models response entries."""
    base_url = get_openrouter_base_url()
    cache = OpenRouterCatalogCache()
    models = await cache.get_models(text_output_only=text_output_only)

    infos: list[dict[str, Any]] = []
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        infos.append(
            {
                "model_id": model_id,
                "name": model_display_name(model),
                "provider": "openrouter",
                "base_url": base_url,
            }
        )

    infos.sort(key=lambda item: item["model_id"])
    return infos
