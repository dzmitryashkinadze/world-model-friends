import polars as pl

from world_model_friends.data_wrangling.script_sequencer import generate_sequences


def test_generate_sequences():
    # Create a dummy dataframe
    data = {
        "Name": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
        "Lines": ["Hello", "Hi there", "How are you?", "I am good", "Fine thanks"],
    }
    df = pl.DataFrame(data)

    num_sequences = 5
    max_context_length = 2

    sequences_df = generate_sequences(df, num_sequences, max_context_length)

    # Check number of rows
    assert len(sequences_df) == num_sequences

    # Check columns
    expected_columns = ["context_names", "context_text", "target_name", "target_text"]
    assert all(col in sequences_df.columns for col in expected_columns)

    # Check some content
    # Since it's random, we check that the columns are non-empty and types are okay
    assert not sequences_df["context_names"].is_null().any()
    assert not sequences_df["target_text"].is_null().any()
