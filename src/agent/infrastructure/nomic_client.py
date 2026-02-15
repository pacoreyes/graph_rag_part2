"""Nomic embedding model manager with dependency injection."""

import asyncio
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class NomicClient:
    """Manager for Nomic embedding model and tokenizer.

    Args:
        model_name: HuggingFace model identifier for Nomic.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize with model name."""
        self._model_name = model_name
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None

    async def get_model(self) -> PreTrainedModel:
        """Get or lazily initialize the Nomic model.

        Returns:
            PreTrainedModel: The Nomic model in eval mode.
        """
        if self._model is None:
            self._model = await asyncio.to_thread(
                AutoModel.from_pretrained, self._model_name, trust_remote_code=True
            )
            self._model.eval()
        return self._model

    async def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get or lazily initialize the Nomic tokenizer.

        Returns:
            PreTrainedTokenizer: The Nomic tokenizer.
        """
        if self._tokenizer is None:
            self._tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, self._model_name, trust_remote_code=True
            )
        return self._tokenizer
