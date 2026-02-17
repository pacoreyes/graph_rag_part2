# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Reusable Google Gemini generation functions.
# Pure functions with dependency injection — no global config or singletons.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import random
import time
from typing import Any

import structlog
from google import genai
from google.genai.errors import ClientError

logger = structlog.get_logger()

MAX_RETRIES = 5
BASE_BACKOFF = 2.0


def gemini_generate(
    client: genai.Client,
    prompt: str,
    system_instruction: str | None = None,
    model: str = "models/gemini-2.0-flash",
    response_mime_type: str = "text/plain",
    response_schema: Any | None = None,
) -> Any:
    """Generate text or structured data using Google Gemini with retries.

    Uses exponential backoff with jitter on 429 (rate limit) errors.
    This function is synchronous — callers should wrap it with
    ``asyncio.to_thread`` to avoid blocking the event loop.

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

    for attempt in range(MAX_RETRIES):
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
            if getattr(e, "code", 0) == 429:
                if attempt == MAX_RETRIES - 1:
                    logger.error(
                        "gemini_rate_limit_exhausted",
                        attempts=MAX_RETRIES,
                    )
                    raise

                sleep_time = BASE_BACKOFF * (2 ** attempt)
                jitter = random.uniform(0, sleep_time * 0.25)
                total_sleep = sleep_time + jitter
                logger.warning(
                    "gemini_rate_limit_retry",
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    sleep_seconds=round(total_sleep, 1),
                )
                time.sleep(total_sleep)
            else:
                raise
        except Exception as e:
            logger.error("gemini_generate_error", error=str(e))
            raise
