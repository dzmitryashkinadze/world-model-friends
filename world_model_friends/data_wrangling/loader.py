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
