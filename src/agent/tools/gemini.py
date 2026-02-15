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
) -> str:
    """Generate text using Google Gemini given a prompt.

    Args:
        client: Gemini client instance (injected).
        prompt: The prompt to send to the model.
        model: Gemini generation model name.

    Returns:
        str: The generated text response.
    """
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
