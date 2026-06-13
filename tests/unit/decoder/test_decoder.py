"""Tests for the decoder module.

Decoder: Takes a predicted semantic embedding and returns the most similar
dialogue lines from the stored Friends script embeddings.
"""

import numpy as np

from world_model_friends.decoder.vector_search_decoder import VectorSearchDecoder


def test_search_returns_top_3_similar_lines() -> None:
    """Given a target embedding, return the 3 most similar lines
    by cosine similarity.
    """
    decoder = VectorSearchDecoder(
        script_with_line_embeddings_path="data/Friends_script_embeddings.parquet",
        top_k=3,
    )
    target = np.zeros(3)  # matches dummy data

    results = decoder.decode(target)
    assert len(results) == 3
    assert all(
        r["cosine_similarity"] >= -1.0 and r["cosine_similarity"] <= 1.0
        for r in results
    )


def test_search_with_character_filter() -> None:
    """Restrict search to lines spoken by a specific character."""
    decoder = VectorSearchDecoder(
        script_with_line_embeddings_path="data/Friends_script_embeddings.parquet",
        top_k=3,
    )
    target = np.zeros(3)
    results = decoder.decode(target, speaker="Alice")
    assert all(r["Name"] == "Alice" for r in results)
    assert len(results) <= 3


def test_search_with_different_embedding() -> None:
    """Test with a non-zero embedding."""
    decoder = VectorSearchDecoder(
        script_with_line_embeddings_path="data/Friends_script_embeddings.parquet",
        top_k=3,
    )
    target = np.array([1.0, 0.5, -0.2])
    results = decoder.decode(target)
    assert len(results) == 3


def test_empty_result_when_no_match() -> None:
    """When filtering by a character that doesn't exist, return empty."""
    decoder = VectorSearchDecoder(
        script_with_line_embeddings_path="data/Friends_script_embeddings.parquet",
        top_k=3,
    )
    target = np.zeros(3)
    results = decoder.decode(target, speaker="NonExistentCharacter")
    assert len(results) == 0


def test_top_k_customization() -> None:
    """Ensure top_k parameter controls the number of results."""
    decoder = VectorSearchDecoder(
        script_with_line_embeddings_path="data/Friends_script_embeddings.parquet",
        top_k=5,
    )
    target = np.zeros(3)
    results = decoder.decode(target)
    assert len(results) <= 5
    assert len(results) <= len(decoder.lines_df)
