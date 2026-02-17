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

import time
import logging
from typing import Any
from google import genai
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

def gemini_generate(
    client: genai.Client,
    prompt: str,
    system_instruction: str | None = None,
    model: str = "models/gemini-2.0-flash",
    # model: str = "models/gemini-2.0-flash-lite",
    response_mime_type: str = "text/plain",
    response_schema: Any | None = None,
) -> Any:
    """Generate text or structured data using Google Gemini with native retries.

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

    max_retries = 5
    backoff = 2.0

    for attempt in range(max_retries):
        try:
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

        except ClientError as e:
            # Check for 429 Resource Exhausted
            if getattr(e, "status_code", 0) == 429:
                if attempt == max_retries - 1:
                    logger.error(f"Gemini API rate limit exceeded after {max_retries} attempts.")
                    raise
                
                sleep_time = backoff * (2 ** attempt)
                logger.warning(f"Gemini 429 Rate Limit. Retrying in {sleep_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                # Re-raise other errors immediately
                raise
        except Exception as e:
            logger.error(f"Unexpected error in gemini_generate: {e}")
            raise
