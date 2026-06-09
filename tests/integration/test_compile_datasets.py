from unittest.mock import MagicMock, patch

import polars as pl

from world_model_friends.data_wrangling.compile_datasets import compile_datasets


def test_compile_datasets_integration(tmp_path):
    # [0] Set up test folder structure
    # We'll use tmp_path which is a pytest fixture for a temporary directory.
    # This satisfies the requirement of a test folder structure and handles cleanup.

    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    test_output_dir = tmp_path / "test_output"
    test_output_dir.mkdir()

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

    # [2] Test logic to convert this synthetic CSV to parquet files
    # We use subprocess to call the CLI as an integration test.
    # We'll use 'uv run' to ensure the environment is correctly set up.

    # To avoid using subprocess and therefore being able to patch the requests call,
    # we use the click CliRunner to run the CLI in the same process.

    with patch("world_model_friends.ai.embeddings.requests.post") as mock_post:

        def side_effect(url, json=None, **kwargs):
            # The payload contains the list of texts in "input"
            texts = json.get("input", []) if json else []
            dim = 3
            data = {"data": [{"embedding": [0.0] * dim} for _ in range(len(texts))]}

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = data
            return mock_response

        mock_post.side_effect = side_effect

        # run the logic
        compile_datasets(
            raw_data_file_path=str(synthetic_csv),
            output_dir=str(test_output_dir),
            num_sequences=10,
            max_context_length=3,
            test_ratio=0.2,
            val_ratio=0.2,
        )

    # [3] Tests on final files
    # Check that parquet files were created in the output directory
    output_files = list(test_output_dir.glob("*.parquet"))
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

    has_train = any("train" in f.name for f in output_files)
    has_test = any("test" in f.name for f in output_files)
    has_val = any("val" in f.name for f in output_files)

    assert has_train
    assert has_test
    assert has_val

    # [4] Clean up
    # tmp_path handles it.
