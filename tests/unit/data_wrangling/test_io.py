import os

import polars as pl
import pytest

from world_model_friends.data_wrangling.io import (
    load_csv_to_polars,
    load_parquet_files,
    save_folds,
)


def test_load_csv_to_polars(tmp_path):
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.write_csv(csv_file)

    # Test loading
    loaded_df = load_csv_to_polars(str(csv_file))
    assert loaded_df.shape == (2, 2)
    assert loaded_df["a"].to_list() == [1, 2]


def test_load_parquet_files(tmp_path):
    # Create temporary parquet files
    p1 = tmp_path / "file1.parquet"
    p2 = tmp_path / "file2.parquet"
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [2]})
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    # Test loading with pattern
    pattern = str(tmp_path / "*.parquet")
    loaded_df = load_parquet_files(pattern)
    assert loaded_df.height == 2
    assert loaded_df["a"].to_list() == [1, 2]

    # Test loading with limit
    loaded_df_limited = load_parquet_files(pattern, limit=1)
    assert loaded_df_limited.height == 1

    # Test ValueError when no files found
    with pytest.raises(ValueError, match="No files found for pattern"):
        load_parquet_files("non_existent_pattern_*.parquet")


def test_save_folds(tmp_path):
    # Create dummy dataframes
    df = pl.DataFrame({"a": [1]})

    # Base path for output (including extension)
    output_base = str(tmp_path / "output.parquet")

    # Test saving
    train_path, test_path, val_path = save_folds(df, df, df, output_base)

    base, ext = os.path.splitext(output_base)
    assert train_path == f"{base}_train{ext}"
    assert test_path == f"{base}_test{ext}"
    assert val_path == f"{base}_val{ext}"

    # Check if files exist and are valid parquet
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    assert os.path.exists(val_path)

    loaded_train = pl.read_parquet(train_path)
    assert loaded_train.height == 1
