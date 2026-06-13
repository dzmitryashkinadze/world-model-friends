from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from world_model_friends.data_wrangling.compile_datasets import compile_datasets


def test_compile_datasets_integration():
    # [0] Set up test folder structure
    # We'll use tmp_path which is a pytest fixture for a temporary directory.
    # This satisfies the requirement of a test folder structure and handles cleanup.

    test_data_dir = Path("tests/data")
    synthetic_csv = test_data_dir / "synthetic.csv"

    # [1] Hard code a synthetic CSV
    csv_content = """Name,Lines
Alice,Hello there!
Bob,"Hi Alice, how are you?"
Alice,"I am doing great, thanks for asking!"
Bob,That is wonderful to hear.
Alice,"So, what are we doing today?"
Bob,We are testing the world model.
Alice,"Oh, that sounds exciting!"
Bob,Indeed it is.
Alice,Let's get started then.
Bob,Agreed!
"""
    synthetic_csv.write_text(csv_content)

    # [2] Mock the embedding model so we don't download from HF,
    # then run the compile logic which internally calls embed_lines
    # and embed_sequences (both use model.encode under the hood)
    #
    # We patch the model reference in the embeddings module because that's
    # where the singleton is actually used (imported from model.py).

    def fake_encode(texts, convert_to_numpy=False):
        # Return fixed-dimension embeddings (match model dim=1024 for qwen3)
        dim = 1024
        return np.zeros((len(texts), dim))

    # Patch the model where it's used, not where it's defined
    with patch("world_model_friends.encoder.embeddings.model.encode", fake_encode):
        # run the logic
        compile_datasets(
            raw_data_file="synthetic.csv",
            data_path=str(test_data_dir),
            n_sequences=10,
            max_context_length=3,
            test_ratio=0.2,
            val_ratio=0.2,
        )

    # [3] Tests on final files
    # Check that parquet files were created in the output directory
    output_files = [
        f"{str(test_data_dir)}/test.parquet",
        f"{str(test_data_dir)}/val.parquet",
        f"{str(test_data_dir)}/train.parquet",
    ]
    assert len(output_files) > 0

    for file_path in output_files:
        df = pl.read_parquet(file_path)
        assert not df.is_empty()
        assert all(
            col in df.columns
            for col in [
                "context_identity",
                "context_embedding",
                "target_identity",
                "target_embedding",
            ]
        )

    # [4] Clean up
    # tmp_path handles it.
