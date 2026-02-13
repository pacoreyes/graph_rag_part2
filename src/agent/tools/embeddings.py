"""Reusable Nomic embedding function.

Pure function with dependency injection — no global config or singletons.
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def nomic_embed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    prefix: str = "search_query: ",
    dimensions: int = 768,
) -> list[float]:
    """Generate an embedding vector using a Nomic model.

    Handles tokenization, forward pass, mean pooling, Matryoshka truncation,
    and L2 normalization.

    Args:
        model: HuggingFace Nomic model instance (injected).
        tokenizer: HuggingFace tokenizer instance (injected).
        text: Text to embed.
        prefix: Task prefix for Nomic models (default for query-time).
        dimensions: Target dimensionality for Matryoshka truncation (default 768).

    Returns:
        list[float]: The L2-normalized embedding vector.
    """
    prefixed = f"{prefix}{text}"
    encoded = tokenizer(prefixed, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pooling over token embeddings, respecting attention mask
    attention_mask = encoded["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    # Matryoshka truncation
    if dimensions < 768:
        mean_pooled = mean_pooled[:, :dimensions]

    # L2 normalize
    normalized = F.normalize(mean_pooled, p=2, dim=1)
    return normalized[0].tolist()
