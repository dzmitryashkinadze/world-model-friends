from unittest.mock import patch

import polars as pl

from world_model_friends.data_wrangling.script_sequencer import (
    embed_sequences,
    generate_sequences,
    split_data,
)


def test_generate_sequences():
    # Create a dummy dataframe
    data = {
        "Name": ["Ross", "Rachel", "Chandler", "Monica", "Joey"],
        "Lines": ["Hello", "Hi there", "How are you?", "I am good", "Fine thanks"],
    }
    df = pl.DataFrame(data)

    num_sequences = 5
    max_context_length = 2

    sequences_df = generate_sequences(df, num_sequences, max_context_length)

    # Check number of rows
    assert len(sequences_df) == num_sequences

    # Check columns
    expected_columns = [
        "context_identity",
        "context_text",
        "target_identity",
        "target_text",
        "context_length",
    ]
    assert all(col in sequences_df.columns for col in expected_columns)

    # Check some content
    # Since it's random, we check that the columns are non-empty and types are okay
    assert not sequences_df["context_identity"].is_null().any()
    assert not sequences_df["target_text"].is_null().any()
    assert (sequences_df["context_length"] >= 1).all() and (
        sequences_df["context_length"] <= max_context_length
    ).all()


def test_prepare_training_data():
    # Mock embed_string to avoid heavy dependencies/actual model loading
    with patch(
        "world_model_friends.data_wrangling.script_sequencer.embed_batch"
    ) as mock_embed:
        mock_emb_val = [0.1, 0.2, 0.3]
        mock_embed.side_effect = lambda model, texts: [mock_emb_val] * len(texts)

        # Create dummy sequences_df
        data = {
            "context_identity": [[1.0, 0.0], [1.0, 0.0]],
            "context_text": ["Alice: Hello\nBob: Hi", "Alice: Hi"],
            "target_identity": [[1.0, 0.0], [1.0, 0.0]],
            "target_text": ["How are you?", "Fine"],
            "context_length": [2, 1],
        }
        sequences_df = pl.DataFrame(data)

        training_df = embed_sequences(sequences_df)

        # Assertions
        assert len(training_df) == 2
        assert training_df.columns == [
            "context_identity",
            "context_embedding",
            "target_identity",
            "target_embedding",
        ]

        # Check embeddings
        assert training_df["context_embedding"].to_list()[0] == mock_emb_val
        assert training_df["target_embedding"].to_list()[0] == mock_emb_val


def test_split_data():
    # Create a dummy dataframe
    data = {
        "id": range(100),
        "value": [i * 0.5 for i in range(100)],
    }
    df = pl.DataFrame(data)

    train_ratio = 0.7
    val_ratio = 0.15

    train_df, val_df, test_df = split_data(
        df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    # Check sum of lengths
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Check approximate lengths
    # 100 * 0.7 = 70
    # 100 * (0.7 + 0.15) = 85 -> 85 - 70 = 15
    # 100 - 85 = 15
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
