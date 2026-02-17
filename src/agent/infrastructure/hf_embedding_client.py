# -----------------------------------------------------------
# GraphRAG system built with Agentic Reasoning
# Hugging Face Inference API embedding client.
#
# (C) 2025-2026 Juan-Francisco Reyes, Cottbus, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
import structlog
from curl_cffi import requests

logger = structlog.get_logger()


class HFEmbeddingClient:
    """Manager for Hugging Face embedding models using HF Inference API (OpenAI compatible).

    Args:
        model_name: HuggingFace model identifier.
        api_token: Hugging Face API token.
    """

    def __init__(self, model_name: str, api_token: str) -> None:
        """Initialize with model name and token."""
        self._model_name = model_name
        self._api_token = api_token
        # Use the new HF Router with explicit pipeline task
        self._api_url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self._model_name}/pipeline/feature-extraction"
        )

    async def embed(
        self, 
        text: str, 
        prefix: str = "Represent this sentence for searching relevant passages: "
    ) -> list[float]:
        """Generate an embedding vector using HF Inference API.

        Args:
            text: Text to embed.
            prefix: Task prefix for retrieval models (e.g., Snowflake).

        Returns:
            list[float]: The raw embedding vector.
        """
        # The new router hf-inference/models route expects 'inputs'
        payload = {
            "inputs": [f"{prefix}{text}"],
        }
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json"
        }
        
        loop = asyncio.get_event_loop()
        
        def _make_request():
            response = requests.post(
                self._api_url, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            if response.status_code != 200:
                logger.error("hf_inference_api_error", status=response.status_code, text=response.text)
                response.raise_for_status()
            return response.json()

        result = await loop.run_in_executor(None, _make_request)
        
        # When sending a list in 'inputs', the router returns a list of lists (vectors)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            if isinstance(result[0], float):
                # Fallback if the API returns a flat list for a single input
                return result
            
        raise ValueError(f"Unexpected response format from HF API: {result}")
