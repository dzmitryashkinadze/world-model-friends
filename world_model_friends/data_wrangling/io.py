"""
Module for handling input/output operations of data files.
"""

import glob

import polars as pl


def load_csv_to_polars(raw_data_file: str, data_path: str) -> pl.DataFrame:
    """
    Loads a CSV file into a Polars DataFrame.

    Args:
        raw_data_file (str): Path to the CSV file.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """
    return pl.read_csv(source=f"{data_path}/{raw_data_file}")


def load_parquet_files(pattern: str, data_path: str) -> pl.DataFrame:
    """
    Loads parquet files matching a pattern into a Polars DataFrame.

    Args:
        pattern (str): Glob pattern for parquet files.
        data_path (str): Path to the data folder.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """
    files = sorted(glob.glob(pathname=f"{data_path}/{pattern}"))
    if not files:
        raise ValueError(f"No files found for pattern: {data_path}/{pattern}")
    return pl.read_parquet(source=files)
