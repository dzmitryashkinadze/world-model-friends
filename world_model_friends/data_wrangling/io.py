"""
Module for handling input/output operations of data files.
"""

import glob
import os

import polars as pl


def load_csv_to_polars(raw_data_file_path: str) -> pl.DataFrame:
    """
    Loads a CSV file into a Polars DataFrame.

    Args:
        raw_data_file_path (str): Path to the CSV file.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """
    return pl.read_csv(source=raw_data_file_path)


def load_parquet_files(pattern: str) -> pl.DataFrame:
    """
    Loads parquet files matching a pattern into a Polars DataFrame.

    Args:
        pattern (str): Glob pattern for parquet files.
        limit (int, optional): Maximum number of files to read.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """
    files = sorted(glob.glob(pathname=pattern))
    if not files:
        raise ValueError(f"No files found for pattern: {pattern}")
    return pl.read_parquet(source=files)


def save_folds(
    train_df: pl.DataFrame, test_df: pl.DataFrame, val_df: pl.DataFrame, output: str
) -> tuple[str, str, str]:
    """
    Saves training, testing, and validation folds as parquet files.

    Args:
        train_df (pl.DataFrame): The training DataFrame.
        test_df (pl.DataFrame): The testing DataFrame.
        val_df (pl.DataFrame): The validation DataFrame.
        output (str): The base output path.

    Returns:
        tuple[str, str, str]: A tuple containing
            (train_output, test_output, val_output).
    """
    base, ext = os.path.splitext(output)
    train_output = f"{base}_train{ext}"
    test_output = f"{base}_test{ext}"
    val_output = f"{base}_val{ext}"

    train_df.write_parquet(file=train_output)
    test_df.write_parquet(file=test_output)
    val_df.write_parquet(file=val_output)

    return train_output, test_output, val_output
