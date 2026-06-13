from unittest.mock import patch

import polars as pl
import pytest

from world_model_friends.data_wrangling.script_sequencer import (
    generate_sequences,
    process_split,
    split_raw_data,
)


@pytest.fixture
def dummy_df():
    return pl.DataFrame({
        "Name": ["Ross", "Rachel", "Chandler", "Monica", "Joey"],
        "Lines": ["Hello", "Hi there", "How are you?", "I am good", "Fine thanks"],
        "line_embedding": [[1.0, 0.0] for _ in range(5)],
    })


@pytest.fixture
def mock_config():
    with patch(
        "world_model_friends.data_wrangling.script_sequencer.get_config"
    ) as mock:
        mock.return_value = ["Ross", "Rachel", "Chandler", "Monica", "Joey"]
        yield mock


def test_generate_sequences(dummy_df):
    n_sequences = 5
    max_context_length = 2

    sequences_df = generate_sequences(dummy_df, n_sequences, max_context_length)

    assert len(sequences_df) == n_sequences
    expected_columns = [
        "context_identity",
        "context_text",
        "target_identity",
        "target_text",
        "target_embedding",
        "context_length",
    ]
    assert all(col in sequences_df.columns for col in expected_columns)
    assert not sequences_df["context_identity"].is_null().any()
    assert not sequences_df["target_text"].is_null().any()
    assert (sequences_df["context_length"] >= 1).all() and (
        sequences_df["context_length"] <= max_context_length
    ).all()


def test_split_raw_data(dummy_df):
    test_ratio = 0.1
    val_ratio = 0.1
    test_df, val_df, train_df = split_raw_data(dummy_df, test_ratio, val_ratio)
    assert len(test_df) + len(val_df) + len(train_df) == len(dummy_df)
    assert len(test_df) == 0
    assert len(val_df) == 1
    assert len(train_df) == 4


def test_process_split(dummy_df):
    with (
        patch(
            "world_model_friends.data_wrangling.script_sequencer.generate_sequences"
        ) as mock_gen,
        patch(
            "world_model_friends.data_wrangling.script_sequencer.embed_sequences"
        ) as mock_emb,
    ):
        mock_gen.return_value = pl.DataFrame({"col": [1]})
        mock_emb.return_value = pl.DataFrame({"col": [1]})

        result = process_split(dummy_df, "train", 5, 2)
        assert result.height == 1
        assert mock_gen.called
        assert mock_emb.called
