"""Integration tests for embeddings."""

from unittest.mock import patch

import numpy as np


def fake_encode(texts, convert_to_numpy=False):
    """Fake encode that returns fixed-dimension embeddings without pulling a model."""
    dim = 1024
    return np.zeros((len(texts), dim))


def test_embed_string_basic():
    """Test the embedding model with a mocked encoder."""
    from world_model_friends.encoder import embed_batch

    text = ["hello world"]

    # Patch the model where it's used, not where it's defined
    with patch("world_model_friends.encoder.embeddings.model.encode", fake_encode):
        embedding = embed_batch(texts=text)

        assert isinstance(embedding, list)
        assert len(embedding[0]) == 1024  # fake_encode dim
        assert all(isinstance(x, float) for x in embedding[0])
