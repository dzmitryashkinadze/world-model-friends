from unittest.mock import MagicMock, patch

import polars as pl

from world_model_friends.data_wrangling.compile_datasets import compile_datasets


def test_compile_datasets_integration(tmp_path):
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    test_output_dir = tmp_path / "test_output"
    test_output_dir.mkdir()

    synthetic_csv = test_data_dir / "synthetic.csv"

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

    # We need to patch get_config to ensure batch_size is not None
    # and patch model.encode to mock the embeddings.
    with (
        patch("world_model_friends.config.get_config") as mock_get_config,
        patch("world_model_friends.encoder.model.model.encode") as mock_encode,
    ):
        # Mock get_config to return sensible values
        def get_config_side_effect(section, key):
            if section == "embeddings" and key == "batch_size":
                return 2
            if section == "embedding" and key == "model_name":
                return "all-MiniLM-L6-v2"
            return None

        mock_get_config.side_effect = get_config_side_effect

        # Mock encode to return a mock object with a .tolist() method
        def encode_side_effect(texts, **kwargs):
            dim = 3
            mock_res = MagicMock()
            mock_res.tolist.return_value = [[0.0] * dim for _ in range(len(texts))]
            return mock_res

        mock_encode.side_effect = encode_side_effect

        # run the logic
        compile_datasets(
            raw_data_file_path=str(synthetic_csv),
            output_dir=str(test_output_dir),
            n_sequences=10,
            max_context_length=3,
            test_ratio=0.2,
            val_ratio=0.2,
        )

    # [3] Tests on final files
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
