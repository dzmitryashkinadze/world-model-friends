import polars as pl
import pytest

from world_model_friends.data_wrangling.io import (
    load_csv_to_polars,
    load_parquet_files,
)


def test_load_csv_to_polars(tmp_path):
    # Create a temporary CSV file
    csv_file = "test.csv"
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.write_csv(f"{tmp_path}/{csv_file}")

    # Test loading
    loaded_df = load_csv_to_polars(str(csv_file), data_path=tmp_path)
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
    pattern = "*.parquet"
    loaded_df = load_parquet_files(pattern, data_path=tmp_path)
    assert loaded_df.height == 2
    assert loaded_df["a"].to_list() == [1, 2]

    # Test ValueError when no files found
    with pytest.raises(ValueError, match="No files found for pattern"):
        load_parquet_files("non_existent_pattern_*.parquet", data_path=tmp_path)
