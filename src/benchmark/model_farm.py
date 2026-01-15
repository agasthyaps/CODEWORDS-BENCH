"""Model farm configuration loader (v1.1).

Loads a list of models and optional matchup strategy from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import ModelConfig


class ModelFarmItem(BaseModel):
    """Single model entry in the farm."""

    model_config = {"protected_namespaces": ()}

    id: str
    short_name: str | None = None
    provider: str = "openrouter"


class ModelFarmFile(BaseModel):
    """Top-level model farm file schema."""

    model_config = {"protected_namespaces": ()}

    model_farm: list[ModelFarmItem] = Field(default_factory=list)
    default_matchups: Literal["round_robin", "subset"] = "round_robin"
    # List of explicit model-id pairs when default_matchups == "subset"
    matchups: list[tuple[str, str]] | None = None
    openrouter_base_url: str | None = None


def load_model_farm(path: str | Path) -> tuple[list[ModelConfig], ModelFarmFile]:
    """
    Load model farm configuration from a JSON file.

    Returns:
        (models, parsed_file)
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    parsed = ModelFarmFile.model_validate(data)

    models: list[ModelConfig] = []
    for item in parsed.model_farm:
        name = item.short_name or (item.id.split("/")[-1] if "/" in item.id else item.id)
        models.append(
            ModelConfig(
                name=name,
                model_id=item.id,
                provider=item.provider,
                base_url=parsed.openrouter_base_url if item.provider == "openrouter" else None,
            )
        )

    return models, parsed

