"""Common parsing utilities for all games."""

from __future__ import annotations

import re


def extract_scratchpad(response: str) -> str | None:
    """Extract SCRATCHPAD: content from a model response.
    
    Looks for a line starting with SCRATCHPAD: and captures everything
    after it until a double newline or end of string.
    
    Args:
        response: The raw model response text
        
    Returns:
        Extracted scratchpad content, or None if not found
        
    Examples:
        >>> extract_scratchpad("CLUE: OCEAN\\nSCRATCHPAD: Remember slot 2 is water-themed")
        'Remember slot 2 is water-themed'
        >>> extract_scratchpad("Just a regular response")
        None
    """
    # Look for SCRATCHPAD: at start of line (case-insensitive)
    # Capture everything until double newline or end of string
    match = re.search(
        r"^SCRATCHPAD\s*:\s*(.+?)(?:\n\n|\Z)",
        response,
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None


def remove_scratchpad_from_response(response: str) -> str:
    """Remove SCRATCHPAD: section from response for cleaner parsing.
    
    Args:
        response: The raw model response text
        
    Returns:
        Response with SCRATCHPAD section removed
    """
    # Remove the SCRATCHPAD line and its content
    cleaned = re.sub(
        r"^SCRATCHPAD\s*:\s*.+?(?:\n\n|\Z)",
        "",
        response,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    return cleaned.strip()
