import os

import polars as pl


def load_csv_to_polars(file_path: str) -> pl.DataFrame:
    """
    Loads a CSV file into a Polars DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pl.DataFrame: The loaded DataFrame.
    """
    return pl.read_csv(file_path)


def save_folds(
    train_df: pl.DataFrame, test_df: pl.DataFrame, val_df: pl.DataFrame, output: str
):
    """
    Saves training, testing, and validation folds as parquet files.

    Args:
        train_df (pl.DataFrame): The training DataFrame.
        test_df (pl.DataFrame): The testing DataFrame.
        val_df (pl.DataFrame): The validation DataFrame.
        output (str): The base output path.

    Returns:
        tuple: (train_output, test_output, val_output)
    """
    base, ext = os.path.splitext(output)
    train_output = f"{base}_train{ext}"
    test_output = f"{base}_test{ext}"
    val_output = f"{base}_val{ext}"

    train_df.write_parquet(train_output)
    test_df.write_parquet(test_output)
    val_df.write_parquet(val_output)

    return train_output, test_output, val_output
