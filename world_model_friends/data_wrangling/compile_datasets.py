import sys

from world_model_friends.config import get_config
from world_model_friends.data_wrangling.io import load_csv_to_polars
from world_model_friends.data_wrangling.script_sequencer import (
    process_split,
    split_raw_data,
)
from world_model_friends.encoder.embeddings import embed_lines


def compile_datasets(
    raw_data_file_path: str,
    output_dir: str,
    num_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
) -> None:
    """
    Reads CSV, splits sequentially, generates, embeds and stores on disk.

    Args:
        raw_data_file_path (str): Path to the CSV file.
        output_dir (str): Directory to save the processed data.
        num_sequences (int): Number of sequences to generate over all splits.
        max_context_length (int): Maximum context length.
        test_ratio (float): Proportion of raw data for testing.
        val_ratio (float): Proportion of raw data for validation.

    Returns:
        None
    """
    try:
        # 1. Load CSV
        # Columns: (Name, Lines)
        df = load_csv_to_polars(raw_data_file_path=raw_data_file_path)
        print(f"Successfully loaded {raw_data_file_path}")

        # 2. Embed lines
        # Columns: (Name, Lines, line_embedding)
        df = embed_lines(df=df)
        df.write_parquet(get_config("process", "script_with_line_embeddings_path"))

        # 3. Split raw data sequentially
        # Columns: (Name, Lines, embedding)
        test_df, val_df, train_df = split_raw_data(
            df=df, test_ratio=test_ratio, val_ratio=val_ratio
        )

        # Distribute the total num_sequences across the splits proportionally
        n_test = int(num_sequences * test_ratio)
        n_val = int(num_sequences * val_ratio)
        n_train = num_sequences - n_test - n_val

        # 4. Generate, Embed and Store for each split
        # processing
        test_df = process_split(
            split_df=test_df,
            split_name="test",
            n_sequences=n_test,
            max_context_length=max_context_length,
            output_dir=output_dir,
        )
        val_df = process_split(
            split_df=val_df,
            split_name="val",
            n_sequences=n_val,
            max_context_length=max_context_length,
            output_dir=output_dir,
        )
        train_df = process_split(
            split_df=train_df,
            split_name="train",
            n_sequences=n_train,
            max_context_length=max_context_length,
            output_dir=output_dir,
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
