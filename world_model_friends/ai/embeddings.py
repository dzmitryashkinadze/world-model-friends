"""
This module provides functionality for generating embeddings
for text using a remote server.
"""

import requests

from world_model_friends.config import get_config


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of strings using a remote llama.cpp server.

    Args:
        texts (list[str]): A list of strings to be embedded.

    Returns:
        list[list[float]]: A list of embeddings,
            where each embedding is a list of floats.
    """
    url = get_config("embeddings", "model_url")
    endpoint = f"{url.rstrip('/')}/v1/embeddings"

    # The OpenAI API format for /v1/embeddings:
    # POST /v1/embeddings
    # { "input": ["string1", "string2"], "model": "..." }
    payload = {
        "input": texts,
    }

    response = requests.post(endpoint, json=payload)
    response.raise_for_status()

    data = response.json()
    # OpenAI response format: { "data": [ {"embedding": [...]}, ... ] }
    return [item["embedding"] for item in data["data"]]
