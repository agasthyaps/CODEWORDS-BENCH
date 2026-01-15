from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    # Remove triple-backtick fences if present.
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_json_object_strict(text: str) -> dict[str, Any] | None:
    """
    Strict JSON parsing policy:
    - Accept only a JSON object (possibly wrapped in a code fence).
    - Do NOT attempt heuristic extraction of embedded JSON beyond stripping fences.
    """
    t = _strip_code_fences(text)
    try:
        obj = json.loads(t)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def parse_code_triple(obj: Any) -> tuple[int, int, int] | None:
    if not isinstance(obj, list) or len(obj) != 3:
        return None
    try:
        digits = tuple(int(x) for x in obj)
    except Exception:
        return None
    if any(d not in (1, 2, 3, 4) for d in digits):
        return None
    if len(set(digits)) != 3:
        return None
    return digits  # type: ignore[return-value]


def parse_confidence_01(obj: Any) -> float | None:
    try:
        x = float(obj)
    except Exception:
        return None
    if not (0.0 <= x <= 1.0):
        return None
    return x


def repair_user_message(kind: str, error: str) -> str:
    """
    Single repair policy: one retry with an explicit JSON-only instruction.
    """
    return (
        f"Your previous output was invalid ({kind} parse error: {error}).\n"
        "Reply with ONLY a valid JSON object matching the required schema. No extra text."
    )


_GUESS_LINE_RE = re.compile(
    r"^\s*GUESS\s*=\s*\(\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*\)\s+CONF\s*=\s*([01](?:\.\d+)?|\.\d+)\s*$",
    re.IGNORECASE,
)


def parse_guess_conf_line(text: str) -> tuple[tuple[int, int, int], float] | None:
    """
    Strict fallback format:
      GUESS=(x,y,z) CONF=p
    where x,y,z are distinct ints in 1..4 and p in [0,1].
    """
    m = _GUESS_LINE_RE.match(text.strip())
    if not m:
        return None
    d1, d2, d3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if len({d1, d2, d3}) != 3:
        return None
    try:
        p = float(m.group(4))
    except Exception:
        return None
    if not (0.0 <= p <= 1.0):
        return None
    return (d1, d2, d3), p


def strict_guess_retry_prompt(error: str) -> str:
    """
    Single retry instruction for guess parsing failures.
    """
    return (
        f"Invalid guess format: {error}\n"
        "Return ONLY this exact format on one line (no JSON, no extra text):\n"
        "GUESS=(x,y,z) CONF=p\n"
        "Constraints: x,y,z are distinct ints in 1..4; p is a number in [0,1]."
    )

