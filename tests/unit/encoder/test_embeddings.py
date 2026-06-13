from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from world_model_friends.encoder.embeddings import (
    embed_batch,
    embed_lines,
    embed_sequences,
)


def test_embed_batch_success():
    """Test embed_batch with a mocked response."""
    with patch("world_model_friends.encoder.embeddings.model.encode") as mock_encode:
        # Mocking the return value of model.encode to have a .tolist() method
        # We use a numpy array as it's what's expected in the real implementation
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        texts = ["text1", "text2"]
        embeddings = embed_batch(texts=texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        mock_encode.assert_called_once_with(texts=texts, convert_to_numpy=True)


def test_embed_batch_failure():
    """Test embed_batch when model.encode fails."""
    with patch("world_model_friends.encoder.embeddings.model.encode") as mock_encode:
        mock_encode.side_effect = Exception("Model error")

        with pytest.raises(Exception) as excinfo:
            embed_batch(texts=["test"])
        assert "Model error" in str(excinfo.value)


def test_embed_lines():
    """Test embed_lines."""
    df = pl.DataFrame({"Lines": ["text1", "text2"]})

    with patch("world_model_friends.encoder.embeddings.model.encode") as mock_encode:
        mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Note: model_name and output_path are unused in current implementation
        result = embed_lines(df)

        assert "line_embedding" in result.columns
        assert result["line_embedding"].to_list()[0] == [0.1, 0.2, 0.3]
        assert result["line_embedding"].to_list()[1] == [0.4, 0.5, 0.6]
        mock_encode.assert_called_once()


def test_embed_sequences():
    sequences_df = pl.DataFrame({
        "context_identity": [[1.0, 0.0], [0.0, 1.0]],
        "context_text": ["Alice: Hello", "Bob: Hi"],
        "target_identity": [[0.0, 1.0], [1.0, 0.0]],
        "target_text": ["Bob: Hi", "Alice: Hello"],
        "target_embedding": [[0.0, 1.0], [1.0, 0.0]],
    })

    # We patch the local imports in embeddings.py
    with (
        patch("world_model_friends.encoder.embeddings.get_config") as mock_cfg,
        patch("world_model_friends.encoder.embeddings.embed_batch") as mock_embed,
        patch("world_model_friends.encoder.embeddings.tqdm", side_effect=lambda x: x),
    ):
        mock_cfg.return_value = 1  # batch_size = 1
        mock_embed.side_effect = lambda texts: [[0.1, 0.1] for _ in texts]

        with patch("polars.DataFrame.write_parquet") as mock_write:
            # Use a dummy data_path
            embed_sequences(sequences_df, split_name="test", data_path=".")

            assert mock_write.called
