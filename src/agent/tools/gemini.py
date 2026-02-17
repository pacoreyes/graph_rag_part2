# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Reusable Google Gemini generation functions.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Reusable Google Gemini generation functions.

Pure functions with dependency injection — no global config or singletons.
"""

from typing import Any
from google import genai


def gemini_generate(
    client: genai.Client,
    prompt: str,
    system_instruction: str | None = None,
    # model: str = "models/gemini-2.0-flash",
    model: str = "models/gemini-2.0-flash-lite",
    response_mime_type: str = "text/plain",
    response_schema: Any | None = None,
) -> Any:
    """Generate text or structured data using Google Gemini.

    Args:
        client: Gemini client instance (injected).
        prompt: User prompt.
        system_instruction: Optional system instruction/persona.
        model: Gemini generation model name.
        response_mime_type: Output MIME type (e.g., "application/json").
        response_schema: Optional Pydantic or JSON schema for structured output.

    Returns:
        Any: Generated content (string or parsed JSON object).
    """
    config: dict[str, Any] = {"response_mime_type": response_mime_type}
    if response_schema:
        config["response_schema"] = response_schema

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            **config
        ),
    )

    if response_mime_type == "application/json":
        return response.parsed
    return response.text
