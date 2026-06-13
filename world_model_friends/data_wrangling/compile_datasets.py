import glob
import os

from world_model_friends.config import get_config
from world_model_friends.data_wrangling.io import load_csv_to_polars, load_parquet_files
from world_model_friends.data_wrangling.script_sequencer import (
    process_split,
    split_raw_data,
)
from world_model_friends.encoder.embeddings import embed_lines


def compile_datasets(
    raw_data_file: str,
    n_sequences: int,
    max_context_length: int,
    test_ratio: float,
    val_ratio: float,
    data_path: str,
) -> None:
    """
    Reads CSV, splits sequentially, generates, embeds and stores on disk.

    Args:
        raw_data_file (str): Path to the CSV file.
        n_sequences (int): Number of sequences to generate over all splits.
        max_context_length (int): Maximum context length.
        test_ratio (float): Proportion of raw data for testing.
        val_ratio (float): Proportion of raw data for validation.
        data_path (str): Data folder path.

    Returns:
        None
    """
    # 1. Load CSV
    # Columns: (Name, Lines)
    df = load_csv_to_polars(
        raw_data_file=raw_data_file,
        data_path=data_path,
    )
    print(f"Successfully loaded {raw_data_file}")

    # 2. Embed lines
    # Columns: (Name, Lines, line_embedding)
    df = embed_lines(df=df)
    df.write_parquet(
        file=f"{data_path}/{
            get_config(section='process', key='script_with_line_embeddings_file')
        }"
    )
    print(f"Successfully embedded {raw_data_file}")

    # 3. Split raw data sequentially
    # Columns: (Name, Lines, embedding)
    test_df, val_df, train_df = split_raw_data(
        df=df, test_ratio=test_ratio, val_ratio=val_ratio
    )

    # Distribute the total n_sequences across the splits proportionally
    n_test = int(n_sequences * test_ratio)
    n_val = int(n_sequences * val_ratio)
    n_train = n_sequences - n_test - n_val

    # 4. Generate, Embed and Store for each split
    # processing
    test_df = process_split(
        split_df=test_df,
        split_name="test",
        n_sequences=n_test,
        max_context_length=max_context_length,
        data_path=data_path,
    )
    val_df = process_split(
        split_df=val_df,
        split_name="val",
        n_sequences=n_val,
        max_context_length=max_context_length,
        data_path=data_path,
    )
    train_df = process_split(
        split_df=train_df,
        split_name="train",
        n_sequences=n_train,
        max_context_length=max_context_length,
        data_path=data_path,
    )

    # Combine all sequences
    load_parquet_files(pattern="train_*.parquet", data_path=data_path).write_parquet(
        file=f"{data_path}/train.parquet"
    )
    load_parquet_files(pattern="val_*.parquet", data_path=data_path).write_parquet(
        file=f"{data_path}/val.parquet"
    )
    load_parquet_files(pattern="test_*.parquet", data_path=data_path).write_parquet(
        file=f"{data_path}/test.parquet"
    )

    # Clean up the embedding chunks
    for f in glob.glob(pathname=f"{data_path}/train_*.parquet"):
        os.remove(path=f)
    for f in glob.glob(pathname=f"{data_path}/test_*.parquet"):
        os.remove(path=f)
    for f in glob.glob(pathname=f"{data_path}/val_*.parquet"):
        os.remove(path=f)
