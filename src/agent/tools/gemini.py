"""Reusable Google Gemini embedding and generation functions.

Pure functions with dependency injection — no global config or singletons.
"""

from google import genai


def gemini_embed(
    client: genai.Client,
    text: str,
    model: str = "gemini-embedding-001",
) -> list[float]:
    """Generate an embedding vector for the given text using Google Gemini.

    Args:
        client: Gemini client instance (injected).
        text: Text to embed.
        model: Gemini embedding model name.

    Returns:
        list[float]: The embedding vector.
    """
    response = client.models.embed_content(model=model, contents=text)
    return response.embeddings[0].values


def gemini_generate(
    client: genai.Client,
    prompt: str,
    model: str = "models/gemini-2.0-flash",
    response_schema: type = None,
    system_instruction: str = None,
) -> str:
    """Generate text or structured data using Google Gemini.

    Args:
        client: Gemini client instance (injected).
        prompt: The prompt to send to the model.
        model: Gemini generation model name.
        response_schema: Optional Pydantic model or schema for structured output.
        system_instruction: Optional system instruction to set context without repeating in prompt.

    Returns:
        str: The generated response text (or JSON string if schema provided).
    """
    config = {}
    if response_schema:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_schema
    if system_instruction:
        config["system_instruction"] = system_instruction

    response = client.models.generate_content(
        model=model, 
        contents=prompt,
        config=config,
    )
    return response.text
